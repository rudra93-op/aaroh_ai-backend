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
# Upgrade pip
# ----------------------------
RUN pip install --upgrade pip

# ----------------------------
# Core numeric stack (ORDER MATTERS)
# ----------------------------
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir scipy==1.9.3

# ----------------------------
# madmom (NO build isolation)
# ----------------------------
RUN pip install --no-cache-dir Cython==3.2.3
RUN pip install --no-cache-dir --no-build-isolation madmom==0.16.1

# Patch madmom for Python 3.11
RUN sed -i "s/from collections import MutableSequence/from collections.abc import MutableSequence/" \
    /usr/local/lib/python3.11/site-packages/madmom/processors.py

# ----------------------------
# Audio + API libs
# ----------------------------
RUN pip install --no-cache-dir \
    librosa==0.11.0 \
    soundfile \
    fastapi \
    uvicorn \
    python-multipart

# ----------------------------
# App code
# ----------------------------
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
