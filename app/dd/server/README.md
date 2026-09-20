# Defect Detection Server

React + FastAPI web interface for defect detection.

## Setup

### Backend

```bash
cd app/dd/server
pip install -r requirements.txt
```

### Frontend

```bash
cd app/dd/server/frontend
npm install
```

## Development

Terminal 1 (Backend):
```bash
cd app/dd/server
uvicorn api.main:app --reload
```

Terminal 2 (Frontend):
```bash
cd app/dd/server/frontend
npm run dev
```

Open http://localhost:5173

## Production

Build frontend:
```bash
cd app/dd/server/frontend
npm run build
```

Run server:
```bash
cd app/dd/server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## API

### POST /api/detect

Upload image for defect detection.

**Request:**
```
Content-Type: multipart/form-data
file: <image binary>
```

**Response:**
```json
{
  "result": "Defect",
  "class_name": "Defect"
}
```

## Architecture

- `api/` - FastAPI backend
  - `main.py` - App entry point, static file serving
  - `engine.py` - Singleton loader for ImgClassifier/DdInferenceWrapper
  - `routes/detect.py` - Detection endpoint
- `frontend/` - React + Vite + TypeScript
  - `src/App.tsx` - Main UI component
