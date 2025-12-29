FROM python:3.11-slim

# system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# upgrade pip FIRST
RUN pip install --upgrade pip

# 🔒 HARD PIN NUMPY & SCIPY (NO NumPy 2 EVER)
RUN pip install \
    numpy==1.23.5 \
    scipy==1.9.3

# Cython required by madmom
RUN pip install Cython==3.2.3

# madmom (compiled against NumPy 1.x)
RUN pip install --no-build-isolation madmom==0.16.1 && \
    sed -i "s/from collections import MutableSequence/from collections.abc import MutableSequence/" \
    /usr/local/lib/python3.11/site-packages/madmom/processors.py

# librosa compatible with NumPy 1.23
RUN pip install librosa==0.10.2.post1

# API dependencies (DO NOT let these upgrade numpy)
RUN pip install \
    fastapi \
    uvicorn \
    python-multipart \
    soundfile

# copy app
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
