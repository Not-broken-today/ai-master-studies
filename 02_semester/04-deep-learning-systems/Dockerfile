FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

RUN git clone https://github.com/cidautai/DarkIR.git /app/DarkIR

COPY requirements.txt .
COPY run_test.py .

RUN mkdir -p assets/inputs assets/results assets/results_Full_1k models

COPY assets/ /app/assets/
COPY models/ /app/models/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app:/app/DarkIR"

CMD ["python", "run_test.py"]