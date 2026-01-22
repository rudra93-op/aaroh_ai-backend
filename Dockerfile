FROM python:3.10-slim

# ---------------- SYSTEM ----------------
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------- PYTHON TOOLING ----------------
RUN pip install --upgrade pip setuptools wheel

# ---------------- NUMERIC STACK (MUST COME FIRST) ----------------
RUN pip install \
    numpy==1.23.5 \
    scipy==1.9.3

# ---------------- CYTHON (MADMOM SAFE) ----------------
RUN pip install Cython==0.29.36

# ---------------- AUDIO LIBS ----------------
RUN pip install \
    librosa==0.10.1 \
    soundfile

# ---------------- MADMOM (SOURCE INSTALL – KEY FIX) ----------------
RUN pip install --no-build-isolation \
    git+https://github.com/CPJKU/madmom.git

# ---------------- API ----------------
RUN pip install \
    fastapi \
    uvicorn \
    python-multipart

# ---------------- APP ----------------
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
