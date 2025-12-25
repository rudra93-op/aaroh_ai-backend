🎸 Aaroh AI – Backend (Chord Recognition API)

Aaroh AI backend is a Dockerized FastAPI service that analyzes uploaded audio files and returns detected guitar chords over time using ML-based audio processing (madmom).

This repository contains only the backend (API + chord engine).

🚀 Live API

Base URL:

https://aaroh-ai-backend.onrender.com


Swagger UI (API Playground):

https://aaroh-ai-backend.onrender.com/docs

📦 Tech Stack

Python 3.11

FastAPI – API framework

madmom – Music information retrieval (chord detection)

NumPy / SciPy / Cython – Audio processing

Docker – Deployment & environment consistency

Render – Cloud hosting

🧠 What This Backend Does

Accepts an audio file (song or guitar recording)

Processes audio features (chroma, harmony)

Detects chords with timestamps

Returns structured chord data as JSON

This backend powers Aaroh’s song-based guitar learning flow.

🗂 Project Structure
backend/
│
├── main.py                # FastAPI app & routes
├── chord_engine.py        # Chord recognition logic
├── Dockerfile             # Docker build instructions
├── .dockerignore          # Docker ignore rules
├── .gitignore             # Git ignore rules
├── requirements.lock.txt  # Locked dependencies (IMPORTANT)
└── requirments.txt        # (legacy / optional)

⚙️ Setup Guide (Local Development)
1️⃣ Prerequisites

Make sure you have installed:

Git

Docker Desktop (Windows / Mac)

WSL (Windows users only)

👉 Docker Desktop must be running.

2️⃣ Clone the Repository
git clone https://github.com/rudra93-op/aaroh_ai-backend.git
cd aaroh_ai-backend

3️⃣ Build Docker Image
docker build -t aaroh-backend .


⏳ First build may take a few minutes (madmom compilation).

4️⃣ Run the Backend
docker run -p 8000:8000 aaroh-backend

5️⃣ Open Swagger UI

Open browser:

http://localhost:8000/docs


You should see:

Swagger UI

POST /analyze endpoint

File upload option

🧪 Testing the API

Open /docs

Expand POST /analyze

Click Try it out

Upload an audio file (.wav, .mp3)

Click Execute

Example Response
[
  {
    "start": 0.0,
    "end": 2.5,
    "chord": "C:maj"
  },
  {
    "start": 2.5,
    "end": 5.0,
    "chord": "G:maj"
  }
]

🐳 Why Docker Is Mandatory

This project uses legacy ML libraries (madmom) that require:

Exact NumPy & Cython versions

Linux build environment

Patched compatibility fixes

👉 Docker guarantees everyone runs the same setup, avoiding dependency issues.

❌ Do NOT try to install dependencies manually with pip install
✅ Always use Docker

☁️ Deployment (Render)

The backend is deployed using Render (Docker runtime).

How deployment works:

Push code to main branch

Render auto-builds Docker image

Service redeploys automatically

No manual server management needed.

🔐 Notes for Team Members

Do not edit dependency versions unless discussed

Do not remove Dockerfile patches

Use /docs for testing

Frontend should call:

POST /analyze
