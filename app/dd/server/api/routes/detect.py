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
