# syntax=docker/dockerfile:1.2
FROM python:3.9-slim

WORKDIR /code


COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]