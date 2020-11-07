FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7 as base

FROM base AS train
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ./src/train.py /src/train.py
COPY ./data/sample_data.csv /data/sample_data.csv
RUN python3 /src/train.py

