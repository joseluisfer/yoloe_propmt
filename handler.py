import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from ultralytics import YOLOE
import cv2

app = FastAPI(title="YOLOE Text Prompt API", version="1.0")

# Variable global para el modelo
model = None

def load_model():
    """Carga el modelo YOLOE-26X una sola vez al iniciar el servidor"""
    global model
    if model is None:
        # Ruta local al archivo .pt (se debe incluir en el build o montar)
        model_path = "yoloe-26x-seg.pt"
        print(f"Cargando modelo desde {model_path}...")
        model = YOLOE(model_path)
        print("Modelo cargado correctamente")
    return model

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] en formato absoluto
    mask: Optional[List[List[float]]] = None  # opcional, segmentación

class PredictResponse(BaseModel):
    detections: List[DetectionResult]

@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar el servicio"""
    load_model()

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    text_prompt: str = Form(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45)
):
    """
    Endpoint para detección con prompt de texto.
    - file: imagen (jpg, png, etc.)
    - text_prompt: clases separadas por coma, ej: "person, car, dog"
    - conf_threshold: umbral de confianza
    - iou_threshold: umbral para NMS
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    # Leer y convertir imagen
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer imagen: {e}")

    # Preparar clases
    classes = [c.strip() for c in text_prompt.split(",") if c.strip()]
    if not classes:
        raise HTTPException(status_code=400, detail="El prompt de texto no puede estar vacío")

    # Establecer clases en el modelo
    model.set_classes(classes)

    # Ejecutar predicción
    results = model.predict(
        source=np.array(pil_image),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Mapear IDs a nombres
            names = result.names
            for i in range(len(boxes)):
                class_name = names.get(cls_ids[i], "unknown")
                detections.append(DetectionResult(
                    class_name=class_name,
                    confidence=float(confs[i]),
                    bbox=boxes[i].tolist()
                ))

    return PredictResponse(detections=detections)

@app.post("/predict_base64", response_model=PredictResponse)
async def predict_base64(
    image_base64: str = Form(...),
    text_prompt: str = Form(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45)
):
    """
    Versión alternativa que recibe la imagen en base64.
    """
    import base64
    try:
        # Eliminar posible cabecera data:image/...
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al decodificar base64: {e}")

    classes = [c.strip() for c in text_prompt.split(",") if c.strip()]
    if not classes:
        raise HTTPException(status_code=400, detail="El prompt de texto no puede estar vacío")

    model.set_classes(classes)

    results = model.predict(
        source=np.array(pil_image),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names
            for i in range(len(boxes)):
                detections.append(DetectionResult(
                    class_name=names.get(cls_ids[i], "unknown"),
                    confidence=float(confs[i]),
                    bbox=boxes[i].tolist()
                ))

    return PredictResponse(detections=detections)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
