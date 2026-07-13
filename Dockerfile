# Immagine base leggera con Python
FROM python:3.11-slim

# Directory di lavoro dentro il container
WORKDIR /app

# Copiamo prima solo requirements.txt: così Docker riusa la cache
# del layer "pip install" finché le dipendenze non cambiano, anche
# se modifichiamo il codice sorgente in seguito.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ora copiamo il resto del codice e i documenti
COPY rag.py app.py ./
COPY documenti/ ./documenti/

# Porta attesa da molti servizi cloud (es. Cloud Run, Hugging Face Spaces)
EXPOSE 8080

# NOTA: la chiave GEMINI_API_KEY NON viene messa nell'immagine.
# Va passata a runtime con -e GEMINI_API_KEY=... oppure tramite
# compose.yaml / secret della piattaforma di deploy.

# 0.0.0.0 = ascolta su tutte le interfacce di rete del container
# (non solo localhost), altrimenti non sarebbe raggiungibile da fuori.
# 8080 = porta standard attesa dai servizi cloud.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8080"]
