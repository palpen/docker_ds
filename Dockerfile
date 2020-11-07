FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7 as base

FROM base AS train

# Install libraries
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY ./src/build_data.py /src/build_data.py
COPY ./src/train.py /src/train.py

# Copy dataset to image
COPY ./data/sample_data.csv /data/sample_data.csv

# Run data preprocessing and train model
RUN python3 /src/build_data.py
RUN python3 /src/train.py

