FROM python:3.12.2-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import AutoModelForTokenClassification, AutoTokenizer; \
               AutoModelForTokenClassification.from_pretrained('cahya/bert-base-indonesian-NER'); \
               AutoTokenizer.from_pretrained('cahya/bert-base-indonesian-NER')"

COPY . .

EXPOSE 8000

CMD uvicorn main:app --host=0.0.0.0 --port=8000