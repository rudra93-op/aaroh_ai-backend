FROM python:3.11-slim

# system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy lock file
COPY requirements.lock.txt .

# upgrade pip
RUN pip install --upgrade pip

# install Cython first (required by madmom)
RUN pip install Cython==3.2.3

# install numpy & scipy explicitly (order matters)
RUN pip install numpy==1.23.5 scipy==1.9.3

# install madmom without build isolation
RUN pip install --no-build-isolation madmom==0.16.1 && \
    sed -i "s/from collections import MutableSequence/from collections.abc import MutableSequence/" \
    /usr/local/lib/python3.11/site-packages/madmom/processors.py


# install remaining deps (if any)
RUN pip install python-multipart fastapi uvicorn

# copy app code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
