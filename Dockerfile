FROM python:3.11-slim

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----------------------------
# Python dependencies
# ----------------------------
COPY requirements.lock.txt .

RUN pip install --upgrade pip

# Install all Python deps in correct order
RUN pip install --no-cache-dir \
    Cython==3.2.3 \
    numpy==1.23.5 \
    scipy==1.9.3 \
    librosa==0.11.0 \
    madmom==0.16.1 \
    fastapi \
    uvicorn \
    python-multipart \
    soundfile

# Patch madmom for Python 3.11
RUN sed -i "s/from collections import MutableSequence/from collections.abc import MutableSequence/" \
    /usr/local/lib/python3.11/site-packages/madmom/processors.py

# ----------------------------
# App code
# ----------------------------
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
