# ⚖️ Chatbot RAG per uno Studio Legale

Assistente conversazionale didattico che risponde a domande sui documenti di uno
studio legale (clausole contrattuali, informativa privacy, regolamento interno),
usando la tecnica RAG (Retrieval-Augmented Generation) con Google Gemini e
un'interfaccia chat in Streamlit.

> ⚠️ Nota etica: è un progetto didattico. Le risposte del chatbot **non
> costituiscono consulenza legale professionale**.

## Come funziona

1. I documenti in `documenti/` vengono caricati e spezzati in chunk (pezzi di testo).
2. Ogni chunk viene trasformato in un embedding (vettore numerico) con Gemini.
3. Alla domanda dell'utente, si calcola il suo embedding e si recuperano (via
   similarità del coseno) i chunk più pertinenti.
4. I chunk recuperati + la domanda vengono inviati a Gemini, che risponde
   **solo in base al contesto fornito** e cita la fonte.

## Requisiti

- Docker e Docker Compose
- Una chiave API Gemini gratuita, ottenibile su [Google AI Studio](https://aistudio.google.com/apikey)
  (nessuna carta di credito richiesta)

## Esecuzione in locale (con Docker Compose)

1. Copia il file di esempio e inserisci la tua chiave:

   ```bash
   cp .env.example .env
   # apri .env e sostituisci con la tua chiave reale:
   # GEMINI_API_KEY=xxxxxxxx
   ```

2. Avvia il container:

   ```bash
   docker compose up --build
   ```

3. Apri il browser su [http://localhost:8080](http://localhost:8080)

## Esecuzione senza Docker (solo Python)

```bash
python -m venv venv
source venv/bin/activate      # su Windows: venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY=xxxxxxxx   # su Windows (PowerShell): $env:GEMINI_API_KEY="xxxxxxxx"
streamlit run app.py
```

## Esempio di domanda/risposta

**Domanda:** "Quanti giorni di preavviso servono per recedere dal contratto?"

**Risposta attesa:** "Il Cliente può recedere dal contratto con un preavviso
scritto di almeno 15 giorni solari, da comunicare tramite PEC o raccomandata A/R.
(Fonte: clausola_recesso.txt)"

## Struttura del progetto

```
.
├── app.py                  # interfaccia chat Streamlit
├── rag.py                  # logica RAG (chunking, embedding, retrieval, generazione)
├── requirements.txt
├── documenti/               # base di conoscenza (file .txt)
│   ├── clausola_recesso.txt
│   ├── informativa_privacy.txt
│   └── regolamento_studio.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
└── .env.example
```

## Sicurezza

- La chiave `GEMINI_API_KEY` **non è mai** scritta nel codice né committata su Git:
  viene letta da variabile d'ambiente / file `.env` (escluso da `.gitignore` e `.dockerignore`).

## Deploy online (gratuito)

### Opzione A — Google Cloud Run

```bash
gcloud run deploy chatbot-rag-studio-legale \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=xxxxxxxx
```

Cloud Run legge il `Dockerfile` presente nella cartella e costruisce l'immagine
automaticamente (build in cloud, senza bisogno di pushare tu stesso l'immagine).
Con `--min-instances 0` (default) il servizio "scala a zero" quando non riceve
traffico: nessuna istanza attiva = nessun costo, restando nel piano gratuito.

### Opzione B — Hugging Face Spaces (SDK "Docker")

1. Crea un nuovo Space, scegliendo SDK = **Docker**.
2. Carica tutti i file del progetto (il Dockerfile verrà rilevato automaticamente).
3. Nelle impostazioni dello Space, aggiungi un **secret** chiamato `GEMINI_API_KEY`
   con la tua chiave (mai nel codice o nel repository).

## Estensioni facoltative implementabili

- Sostituire il retrieval "a mano" con un vector DB (Chroma/FAISS)
- Chunking per frasi/paragrafi invece che per numero di parole
- Pannello "mostra i passaggi recuperati" (già incluso come expander in app.py)
- Memoria conversazionale per domande di follow-up
- Upload di documenti da UI con `st.file_uploader`
- Secret Manager su Cloud Run al posto di `--set-env-vars`
