FROM python:3.10-slim

# ---------------- SYSTEM DEPENDENCIES ----------------
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------- PYTHON TOOLING ----------------
RUN pip install --upgrade pip setuptools wheel

# ---------------- NUMERIC STACK (PINNED) ----------------
RUN pip install \
    numpy==1.23.5 \
    scipy==1.9.3

# ---------------- BUILD TOOLS ----------------
RUN pip install Cython==0.29.36

# ---------------- AUDIO + ML ----------------
RUN pip install \
    librosa==0.10.1 \
    soundfile \
    madmom==0.16.1

# ---------------- API ----------------
RUN pip install \
    fastapi \
    uvicorn \
    python-multipart

# ---------------- APP CODE ----------------
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
