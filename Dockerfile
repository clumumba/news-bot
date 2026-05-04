FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY data ./data
COPY sample_data ./sample_data
COPY src ./src

ENV PYTHONPATH=/app/src

RUN python -m nlp_pipeline.main train --config configs/pipeline.yaml

EXPOSE 8000

CMD ["uvicorn", "nlp_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
