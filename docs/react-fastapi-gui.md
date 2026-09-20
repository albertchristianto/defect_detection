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
