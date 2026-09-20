# React + FastAPI GUI Design

**Date:** 2026-01-20  
**Status:** Approved

## Purpose

Replace the existing PyQt UI (`app/dd/ui`) with a web-based GUI using React (frontend) and FastAPI (backend), maintaining all existing features while enabling future Triton server integration.

## Background

Current state:
- PyQt5 desktop UI in `app/dd/ui/`
- Two backends: Python (`ImgClassifier` with ONNX Runtime) and C++ (`DdInferenceWrapper` via DLL/SO)
- Config-driven engine selection via `cfgs/AI_System.cfg`
- Features: image upload, display, defect detection, result visualization (colored border)

Project constraints:
- Backend code (`app/dd/lib/`) cannot be modified
- Solutions go in `server/` (new directory)
- Triton server support will be added later

## Goals

1. Reuse existing backend classes without modification
2. Match existing UI functionality
3. Single-process deployment (no reverse proxy, Docker, or complex setup)
4. Prepare structure for future Triton integration

## Architecture

### Monorepo Structure

```
server/
  api/
    __init__.py
    main.py              # FastAPI app entry point
    routes/
      __init__.py
      detect.py          # POST /api/detect endpoint
    engine.py            # Engine loader singleton
  frontend/
    index.html
    src/
      App.tsx            # Main component
      App.css
      main.tsx
    vite.config.ts
    package.json
    tsconfig.json
  requirements.txt       # FastAPI, uvicorn, python-multipart
```

### Backend (FastAPI)

**`server/api/main.py`**
- FastAPI app with CORS middleware
- Mounts `server/frontend/dist/` as static files at `/` (production)
- Includes `/api` router

**`server/api/engine.py`**
- Reads `cfgs/AI_System.cfg` (relative to repo root)
- Instantiates `ImgClassifier` or `DdInferenceWrapper` at startup
- Singleton pattern — one engine per process
- Extensibility point for Triton client later

**`server/api/routes/detect.py`**
- `POST /api/detect` — accepts `multipart/form-data` image
- Decodes image with OpenCV/NumPy
- Calls `engine.forward(image)`
- Returns `{ "result": str, "class_name": str }` as JSON

### Frontend (React + Vite + TypeScript)

**`server/frontend/src/App.tsx`**
- Single component matching PyQt window:
  1. Clickable image area (opens file picker) or drag-drop
  2. Image preview (resized to fit container)
  3. "Detect" button
  4. Result overlay: colored border (red = Defect, green = OK/Normal) + class name label

**`server/frontend/vite.config.ts`**
- Proxy `/api/*` to `http://localhost:8000` in dev mode

**Styling (`App.css`)**
- 640×640 main container (matching PyQt window size)
- Image display area with click/drop handlers
- Detect button below image
- Result border overlay

## API Contract

### POST /api/detect

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

**Error response:**
```json
{
  "detail": "Engine not ready" | "Invalid image" | etc.
}
```

## Development Workflow

**Terminal 1 (Backend):**
```bash
cd server
uvicorn api.main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd server/frontend
npm run dev
```

Vite dev server proxies `/api/*` → FastAPI backend.

## Production Deployment

1. Build frontend: `cd server/frontend && npm run build`
2. Start server: `uvicorn api.main:app` (serves static files from `frontend/dist/`)

Single process, single port.

## Testing Strategy

- **Backend**: Test `/api/detect` with sample images from `app/samples/`
- **Frontend**: Manual testing — upload sample images, verify detection and visualization
- **Integration**: End-to-end test with both Python and C++ backends

## Future Extensions

- **Triton server support**: Add Triton client in `engine.py` as a third backend option
- **Batch processing**: Multi-image upload/detection
- **Real-time inference**: WebSocket streaming for video/camera input

## Non-Goals

- Authentication/authorization (future work)
- Image history/database storage
- Advanced UI features beyond current PyQt functionality
- Deployment automation (Docker, CI/CD)
