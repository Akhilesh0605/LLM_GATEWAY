FROM python:3.12-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#downloading model at building 
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
#copy project

COPY . .

#start the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]