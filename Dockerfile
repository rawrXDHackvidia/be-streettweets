FROM python:3.13.3-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8000

# RUN curl -fsSL https://ollama.com/install.sh | sh

CMD uvicorn main:app --host=0.0.0.0 --port=8000