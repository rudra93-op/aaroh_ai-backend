FROM python:3.11-slim

# 🚫 stop pip from upgrading deps silently
ENV PIP_NO_DEPENDENCY_RESOLUTION=1

# system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# upgrade pip
RUN pip install --upgrade pip

# 🔒 NUMPY / SCIPY (LOCKED)
RUN pip install \
    numpy==1.23.5 \
    scipy==1.9.3 \
    Cython==3.2.3

# madmom (compiled against numpy 1.x)
RUN pip install --no-build-isolation madmom==0.16.1 && \
    sed -i "s/from collections import MutableSequence/from collections.abc import MutableSequence/" \
    /usr/local/lib/python3.11/site-packages/madmom/processors.py

# SAFE librosa (does NOT pull numpy 2)
RUN pip install librosa==0.10.2.post1

# API deps
RUN pip install \
    fastapi \
    uvicorn \
    python-multipart \
    soundfile

# app code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
