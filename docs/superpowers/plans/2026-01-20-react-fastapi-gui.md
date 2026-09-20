# React + FastAPI GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PyQt UI with React frontend and FastAPI backend, reusing existing inference engines.

**Architecture:** FastAPI wraps existing `ImgClassifier`/`DdInferenceWrapper` via singleton engine loader; React frontend calls `/api/detect` for inference; FastAPI serves built React app as static files.

**Tech Stack:** FastAPI, Uvicorn, React, Vite, TypeScript, OpenCV, NumPy

**Spec:** `docs/superpowers/specs/2026-01-20-react-fastapi-gui-design.md`

## Global Constraints

- Python 3.7+ (match existing codebase)
- No modifications to `app/dd/lib/` backend code
- Config path: `cfgs/AI_System.cfg` (relative to repo root)
- Window size: 640×640px (match PyQt UI)

---

## Task 1: Backend Foundation - FastAPI App and Engine Loader

**Files:**
- Create: `server/api/__init__.py`
- Create: `server/api/main.py`
- Create: `server/api/engine.py`
- Create: `server/requirements.txt`

**Interfaces:**
- Consumes: `app/dd/lib/py/ImgClassifier.ImgClassifier`, `app/dd/lib/cpp/DdInference.DdInferenceWrapper`, `cfgs/AI_System.cfg`
- Produces: `app: FastAPI` instance, `get_engine() -> ImgClassifier | DdInferenceWrapper`

- [ ] **Step 1: Write the failing test for engine loader**

Create `server/tests/test_engine.py`:
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../app')))
from server.api.engine import get_engine

def test_engine_loads():
    engine = get_engine()
    assert engine is not None
    assert hasattr(engine, 'forward')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_engine.py::test_engine_loads -v`
Expected: FAIL with "No module named 'server.api.engine'"

- [ ] **Step 3: Create server directory structure and requirements**

Create `server/requirements.txt`:
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
opencv-python==4.8.1.78
numpy==1.24.3
```

Create `server/api/__init__.py` (empty file)

Create `server/tests/__init__.py` (empty file)

- [ ] **Step 4: Write minimal engine loader implementation**

Create `server/api/engine.py`:
```python
import sys
import os
import json
from pathlib import Path

# Add app directory to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'app'))

from dd.lib.py.ImgClassifier import ImgClassifier
from dd.lib.cpp.DdInference import DdInferenceWrapper

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        cfg_path = repo_root / 'cfgs' / 'AI_System.cfg'
        with open(cfg_path) as f:
            config = json.load(f)
        
        if config['mode'] == 'ImgClassifier':
            if config['backend'] == 'py':
                _engine = ImgClassifier(str(repo_root / config['img_classifier_cfg_path']))
            elif config['backend'] == 'cpp':
                _engine = DdInferenceWrapper()
    return _engine
```

- [ ] **Step 5: Write minimal FastAPI app**

Create `server/api/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_engine.py::test_engine_loads -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add server/
git commit -m "feat: add FastAPI foundation and engine loader"
```

---

## Task 2: Detection API Endpoint

**Files:**
- Create: `server/api/routes/__init__.py`
- Create: `server/api/routes/detect.py`
- Modify: `server/api/main.py`

**Interfaces:**
- Consumes: `get_engine() -> ImgClassifier | DdInferenceWrapper` from Task 1
- Produces: `POST /api/detect` endpoint accepting multipart/form-data, returning `{"result": str, "class_name": str}`

- [ ] **Step 1: Write the failing test for detect endpoint**

Create `server/tests/test_detect.py`:
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../app')))
from fastapi.testclient import TestClient
from server.api.main import app

client = TestClient(app)

def test_detect_endpoint_no_file():
    response = client.post("/api/detect")
    assert response.status_code == 422

def test_detect_endpoint_with_image():
    # Use a sample image from the repo
    sample_path = os.path.join(os.path.dirname(__file__), '../../app/samples/mt_normal.jpg')
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(__file__), '../../app/samples/mt_defect.jpg')
    
    with open(sample_path, 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        response = client.post("/api/detect", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert 'result' in data
    assert 'class_name' in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_detect.py -v`
Expected: FAIL with "404 Not Found"

- [ ] **Step 3: Write detect route implementation**

Create `server/api/routes/__init__.py` (empty file)

Create `server/api/routes/detect.py`:
```python
from fastapi import APIRouter, File, UploadFile, HTTPException
import cv2
import numpy as np
from ..engine import get_engine

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Read image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Run inference
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not ready")
        
        result = engine.forward(img)
        
        return {
            "result": result,
            "class_name": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: Integrate route into main app**

Modify `server/api/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import detect

app = FastAPI(title="Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd server && python -m pytest tests/test_detect.py -v`
Expected: PASS

- [ ] **Step 6: Manual test with curl**

Run backend:
```bash
cd server
uvicorn api.main:app --reload
```

In another terminal:
```bash
curl -X POST http://localhost:8000/api/detect \
  -F "file=@../app/samples/mt_normal.jpg"
```

Expected: JSON response with `{"result": "...", "class_name": "..."}`

- [ ] **Step 7: Commit**

```bash
git add server/api/routes/ server/api/main.py server/tests/test_detect.py
git commit -m "feat: add detection API endpoint"
```

---

## Task 3: React Frontend Setup

**Files:**
- Create: `server/frontend/package.json`
- Create: `server/frontend/tsconfig.json`
- Create: `server/frontend/vite.config.ts`
- Create: `server/frontend/index.html`
- Create: `server/frontend/src/main.tsx`
- Create: `server/frontend/src/vite-env.d.ts`

**Interfaces:**
- Consumes: Nothing (standalone setup)
- Produces: Vite dev server on port 5173, proxying `/api/*` to `localhost:8000`

- [ ] **Step 1: Create frontend directory and package.json**

Create `server/frontend/package.json`:
```json
{
  "name": "defect-detection-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.3.3",
    "vite": "^5.0.8"
  }
}
```

- [ ] **Step 2: Create TypeScript config**

Create `server/frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

Create `server/frontend/tsconfig.node.json`:
```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 3: Create Vite config with API proxy**

Create `server/frontend/vite.config.ts`:
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

- [ ] **Step 4: Create HTML entry point**

Create `server/frontend/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Defect Detection</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Create React entry point**

Create `server/frontend/src/vite-env.d.ts`:
```typescript
/// <reference types="vite/client" />
```

Create `server/frontend/src/main.tsx`:
```typescript
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

- [ ] **Step 6: Install dependencies**

Run:
```bash
cd server/frontend
npm install
```

- [ ] **Step 7: Verify dev server starts (with placeholder App)**

Create temporary `server/frontend/src/App.tsx`:
```typescript
export default function App() {
  return <div>Hello World</div>
}
```

Create temporary `server/frontend/src/App.css` (empty file)

Run:
```bash
cd server/frontend
npm run dev
```

Expected: Server starts on `http://localhost:5173`, displays "Hello World"

- [ ] **Step 8: Commit**

```bash
git add server/frontend/
git commit -m "feat: add React + Vite frontend setup"
```

---

## Task 4: React UI Component

**Files:**
- Modify: `server/frontend/src/App.tsx`
- Modify: `server/frontend/src/App.css`

**Interfaces:**
- Consumes: `POST /api/detect` from Task 2
- Produces: Complete UI matching PyQt functionality (image upload, display, detect, result overlay)

- [ ] **Step 1: Write App component with image upload**

Replace `server/frontend/src/App.tsx`:
```typescript
import { useState } from 'react'

export default function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [result, setResult] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      setSelectedImage(URL.createObjectURL(file))
      setResult(null)
    }
  }

  const handleImageClick = () => {
    document.getElementById('file-input')?.click()
  }

  const handleDetect = async () => {
    if (!imageFile) return
    
    setLoading(true)
    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setResult(data.result)
    } catch (error) {
      console.error('Detection failed:', error)
      alert('Detection failed')
    } finally {
      setLoading(false)
    }
  }

  const getBorderColor = () => {
    if (!result) return 'transparent'
    return result === 'Defect' ? 'red' : 'green'
  }

  return (
    <div className="app">
      <div className="container">
        <div 
          className="image-area"
          onClick={handleImageClick}
          style={{ borderColor: getBorderColor() }}
        >
          {selectedImage ? (
            <img src={selectedImage} alt="Selected" />
          ) : (
            <div className="placeholder">Click to select image</div>
          )}
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            style={{ display: 'none' }}
          />
        </div>
        <button 
          className="detect-button"
          onClick={handleDetect}
          disabled={!imageFile || loading}
        >
          {loading ? 'Detecting...' : 'Detect'}
        </button>
        {result && (
          <div className="result">
            Result: {result}
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Write CSS styling**

Replace `server/frontend/src/App.css`:
```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background-color: #f0f0f0;
}

.app {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
}

.container {
  width: 640px;
  height: 640px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  padding: 10px;
  display: flex;
  flex-direction: column;
}

.image-area {
  flex: 1;
  border: 15px solid transparent;
  cursor: pointer;
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
  transition: border-color 0.3s;
}

.image-area img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.placeholder {
  color: #999;
  font-size: 18px;
  text-align: center;
}

.detect-button {
  width: 100%;
  padding: 12px;
  margin-top: 10px;
  font-size: 16px;
  font-weight: 500;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.detect-button:hover:not(:disabled) {
  background-color: #0056b3;
}

.detect-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.result {
  margin-top: 10px;
  padding: 8px;
  text-align: center;
  font-size: 16px;
  font-weight: 500;
}
```

- [ ] **Step 3: Test frontend manually**

Terminal 1:
```bash
cd server
uvicorn api.main:app --reload
```

Terminal 2:
```bash
cd server/frontend
npm run dev
```

Test:
1. Open `http://localhost:5173`
2. Click image area → select image from `app/samples/`
3. Image displays
4. Click "Detect" button
5. Border turns red (Defect) or green (Normal)
6. Result text displays

- [ ] **Step 4: Commit**

```bash
git add server/frontend/src/App.tsx server/frontend/src/App.css
git commit -m "feat: add React UI component with detection"
```

---

## Task 5: Production Build and Static File Serving

**Files:**
- Modify: `server/api/main.py`
- Create: `server/frontend/.gitignore`

**Interfaces:**
- Consumes: Built React app from `server/frontend/dist/`
- Produces: FastAPI serves static files at `/`, single-process deployment

- [ ] **Step 1: Add .gitignore for frontend**

Create `server/frontend/.gitignore`:
```
node_modules
dist
.DS_Store
```

- [ ] **Step 2: Update FastAPI to serve static files**

Modify `server/api/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .routes import detect

app = FastAPI(title="Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok"}

# Serve static files (production build)
static_dir = Path(__file__).parent.parent / "frontend" / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
```

- [ ] **Step 3: Build frontend**

Run:
```bash
cd server/frontend
npm run build
```

Expected: Creates `server/frontend/dist/` directory

- [ ] **Step 4: Test production deployment**

Run:
```bash
cd server
uvicorn api.main:app
```

Open browser to `http://localhost:8000`
Expected: React app loads, all functionality works

- [ ] **Step 5: Commit**

```bash
git add server/api/main.py server/frontend/.gitignore
git commit -m "feat: add static file serving for production build"
```

---

## Task 6: Documentation

**Files:**
- Create: `server/README.md`
- Create: `docs/react-fastapi-gui.md`

**Interfaces:**
- Consumes: Nothing
- Produces: User documentation and developer setup guide

- [ ] **Step 1: Write server README**

Create `server/README.md`:
```markdown
# Defect Detection Server

React + FastAPI web interface for defect detection.

## Setup

### Backend

```bash
cd server
pip install -r requirements.txt
```

### Frontend

```bash
cd server/frontend
npm install
```

## Development

Terminal 1 (Backend):
```bash
cd server
uvicorn api.main:app --reload
```

Terminal 2 (Frontend):
```bash
cd server/frontend
npm run dev
```

Open http://localhost:5173

## Production

Build frontend:
```bash
cd server/frontend
npm run build
```

Run server:
```bash
cd server
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
```

- [ ] **Step 2: Write user documentation**

Create `docs/react-fastapi-gui.md`:
```markdown
# React + FastAPI GUI

Web-based interface for defect detection, replacing the PyQt desktop UI.

## Features

- Image upload (click or drag-drop)
- Real-time defect detection
- Visual result overlay (red border = Defect, green = Normal)
- Supports both Python and C++ backends

## Usage

1. Start the server (see `server/README.md`)
2. Open the web interface
3. Click the image area to select an image
4. Click "Detect" to run inference
5. Result displays as colored border + text label

## Configuration

Uses `cfgs/AI_System.cfg` to select backend:
- `backend: "py"` - Python ONNX Runtime
- `backend: "cpp"` - C++ DLL/SO

## Future Extensions

- Triton server support
- Batch image processing
- Real-time video inference
```

- [ ] **Step 3: Commit**

```bash
git add server/README.md docs/react-fastapi-gui.md
git commit -m "docs: add server and GUI documentation"
```

---

## Verification

After all tasks complete:

1. **Backend tests pass:**
   ```bash
   cd server
   python -m pytest tests/ -v
   ```

2. **Frontend builds without errors:**
   ```bash
   cd server/frontend
   npm run build
   ```

3. **End-to-end manual test:**
   - Start production server: `cd server && uvicorn api.main:app`
   - Open `http://localhost:8000`
   - Upload `app/samples/mt_normal.jpg` → green border
   - Upload `app/samples/mt_defect.jpg` → red border (if exists)

4. **Test both backends:**
   - Edit `cfgs/AI_System.cfg` to switch `backend: "py"` / `backend: "cpp"`
   - Restart server, verify detection works

All tests pass → implementation complete.
