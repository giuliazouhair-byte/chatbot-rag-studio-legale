FROM python:3.11-slim


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY rag.py app.py ./
COPY documenti/ ./documenti/
COPY .streamlit/ ./.streamlit/


EXPOSE 8080



CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8080"]
