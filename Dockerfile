FROM python:3.11-slim

# ---------------- SYSTEM DEPS ----------------
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------- PYTHON TOOLING ----------------
RUN pip install --upgrade pip setuptools wheel

# ---------------- CORE NUMERIC STACK (PINNED) ----------------
# MUST be installed BEFORE madmom
RUN pip install \
    numpy==1.23.5 \
    scipy==1.9.3

# ---------------- BUILD TOOLS ----------------
RUN pip install Cython==3.2.3

# ---------------- AUDIO + ML ----------------
RUN pip install \
    librosa==0.11.0 \
    soundfile \
    madmom==0.16.1

# ---- madmom Python 3.11 FIX ----
RUN sed -i "s/from collections import MutableSequence/from collections.abc import MutableSequence/" \
    /usr/local/lib/python3.11/site-packages/madmom/processors.py

# ---------------- API ----------------
RUN pip install \
    fastapi \
    uvicorn \
    python-multipart

# ---------------- APP CODE ----------------
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
