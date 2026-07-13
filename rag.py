"""
rag.py
Logica RAG (Retrieval-Augmented Generation) per il chatbot dello studio legale.
 
Pipeline:
1) Caricamento documenti e chunking
2) Costruzione indice con embedding (Gemini)
3) Retrieval dei chunk più rilevanti (similarità del coseno)
4) Generazione della risposta con Gemini, basata solo sul contesto recuperato
"""
 
import os
import glob
import numpy as np
from google import genai
from google.genai import types
 
# ----------------------------------------------------------------------
# Configurazione client Gemini
# La chiave API viene letta da variabile d'ambiente, MAI scritta nel codice.
# ----------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY non impostata. Esporta la variabile d'ambiente "
        "prima di avviare l'app (mai scrivere la chiave nel codice)."
    )
 
client = genai.Client(api_key=GEMINI_API_KEY)
 
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.1-flash-lite"
 
DOCS_FOLDER = "documenti"
CHUNK_SIZE_WORDS = 150      # parole per chunk
CHUNK_OVERLAP_WORDS = 30    # sovrapposizione tra chunk consecutivi
TOP_K = 3                   # numero di passaggi da recuperare
 
 
# ----------------------------------------------------------------------
# 1) DOCUMENTI E CHUNKING
# ----------------------------------------------------------------------
def load_documents(folder: str = DOCS_FOLDER) -> list[dict]:
    """
    Carica tutti i file .txt dalla cartella indicata.
    Ritorna una lista di dict: {"source": nome_file, "text": contenuto}
    """
    documents = []
    filepaths = sorted(glob.glob(os.path.join(folder, "*.txt")))
 
    if not filepaths:
        raise FileNotFoundError(
            f"Nessun file .txt trovato nella cartella '{folder}/'. "
            "Aggiungi almeno un documento di esempio."
        )
 
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"source": os.path.basename(path), "text": text})
 
    return documents
 
 
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """
    Spezza un testo lungo in chunk più piccoli, basati sul numero di parole,
    con una sovrapposizione tra un chunk e il successivo (per non perdere
    contesto ai bordi).
    """
    words = text.split()
    if not words:
        return []
 
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  # arretra per creare l'overlap
 
    return chunks
 
 
def build_chunks(documents: list[dict]) -> list[dict]:
    """
    Applica il chunking a tutti i documenti caricati.
    Ritorna una lista di dict: {"text": chunk, "source": nome_file}
    """
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for piece in pieces:
            all_chunks.append({"text": piece, "source": doc["source"]})
    return all_chunks
 
 
# ----------------------------------------------------------------------
# 2) EMBEDDING E COSTRUZIONE INDICE
# ----------------------------------------------------------------------
def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """
    Calcola l'embedding (vettore numerico) di un testo usando Gemini.
    task_type distingue tra embedding di documenti ("RETRIEVAL_DOCUMENT")
    ed embedding di query ("RETRIEVAL_QUERY"), per risultati più accurati.
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)
 
 
def build_index(chunks: list[dict]) -> list[dict]:
    """
    Costruisce l'indice vettoriale: per ogni chunk calcola l'embedding
    e lo aggiunge come nuovo campo. Ritorna la lista arricchita:
    {"text": ..., "source": ..., "embedding": np.ndarray}
    """
    index = []
    for chunk in chunks:
        vector = embed_text(chunk["text"], task_type="RETRIEVAL_DOCUMENT")
        index.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "embedding": vector,
        })
    return index
 
 
def create_index_from_scratch(folder: str = DOCS_FOLDER) -> list[dict]:
    """Funzione di comodo: carica i documenti, li spezza e costruisce l'indice."""
    documents = load_documents(folder)
    chunks = build_chunks(documents)
    index = build_index(chunks)
    return index
 
 
# ----------------------------------------------------------------------
# 3) RETRIEVAL (similarità del coseno)
# ----------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcola la similarità del coseno tra due vettori."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
 
 
def retrieve(query: str, index: list[dict], k: int = TOP_K) -> list[dict]:
    """
    Data una domanda, calcola il suo embedding e lo confronta con tutti
    i vettori dell'indice tramite similarità del coseno.
    Ritorna i k chunk più simili, ordinati per rilevanza decrescente.
    """
    query_vector = embed_text(query, task_type="RETRIEVAL_QUERY")
 
    scored = []
    for item in index:
        score = cosine_similarity(query_vector, item["embedding"])
        scored.append({**item, "score": score})
 
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]
 
 
# ----------------------------------------------------------------------
# 4) GENERAZIONE (prompt aumentato + Gemini)
# ----------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = """Sei un assistente didattico per uno studio legale.
Rispondi ESCLUSIVAMENTE sulla base del contesto fornito qui sotto, estratto dai documenti
dello studio. Non inventare informazioni che non sono presenti nel contesto.
 
Regole:
- Se la risposta si trova nel contesto, rispondi in modo chiaro e cita sempre la fonte
  (il nome del file) tra parentesi alla fine della frase pertinente.
- Se la risposta NON si trova nel contesto, dichiara esplicitamente che l'informazione
  non è presente nei documenti a disposizione. Non provare a indovinare.
- Ricorda che sei un assistente didattico: le tue risposte non costituiscono consulenza
  legale professionale.
"""
 
 
def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Costruisce il prompt aumentato unendo contesto recuperato e domanda."""
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[Fonte: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)
 
    prompt = f"""{SYSTEM_INSTRUCTIONS}
 
CONTESTO:
{context}
 
DOMANDA DELL'UTENTE:
{query}
 
RISPOSTA:"""
    return prompt
 
 
def generate_answer(query: str, index: list[dict], k: int = TOP_K) -> dict:
    """
    Pipeline completa di risposta:
    1) recupera i chunk rilevanti
    2) costruisce il prompt aumentato
    3) chiama Gemini per generare la risposta
 
    Ritorna {"answer": str, "sources": list[str], "chunks": list[dict]}
    """
    retrieved_chunks = retrieve(query, index, k=k)
    prompt = build_prompt(query, retrieved_chunks)
 
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
 
    sources = sorted(set(chunk["source"] for chunk in retrieved_chunks))
 
    return {
        "answer": response.text,
        "sources": sources,
        "chunks": retrieved_chunks,
    }