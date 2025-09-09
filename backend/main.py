# main.py
import os
import uuid
import shutil
import datetime
import logging

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from model.yolo_model import is_cow_in_image
from model.landmark_model import detect_landmarks_and_scale
from utils.calculations import compute_measurements_and_score
from utils.json_handler import save_result, read_results

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-atc-backend")

app = FastAPI(title="AI-ATC Backend (Prototype)", version="0.1.0")

# CORS (adjust allow_origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
def root():
    return {"status": "ok", "message": "AI-ATC backend running"}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Basic validation
    if not file.content_type.startswith("image/"):
        return JSONResponse({"status": "error", "message": "Invalid file type"}, status_code=400)

    # Save uploaded file
    filename = f"{uuid.uuid4().hex}_{file.filename.replace(' ', '_')}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.exception("Failed saving uploaded file")
        return JSONResponse({"status": "error", "message": f"File save error: {e}"}, status_code=500)

    # 1) Detection: is there a cow/buffalo?
    try:
        found, detections = is_cow_in_image(file_path)
    except Exception as e:
        logger.exception("YOLO detection error")
        return JSONResponse({"status": "error", "message": f"Detection error: {e}"}, status_code=500)

    if not found:
        return JSONResponse({"status": "error", "message": "No cow/buffalo detected in image"}, status_code=400)

    # 2) Landmark detection + optional scale detection (returns dict of landmarks and scale factor)
    try:
        landmarks, scale_cm_per_pixel = detect_landmarks_and_scale(file_path)
    except Exception as e:
        logger.exception("Landmark detection error")
        return JSONResponse({"status": "error", "message": f"Landmark detection error: {e}"}, status_code=500)

    # 3) Measurements and score
    try:
        measurements = compute_measurements_and_score(
            landmarks, scale_cm_per_pixel)
    except Exception as e:
        logger.exception("Calculation error")
        return JSONResponse({"status": "error", "message": f"Computation error: {e}"}, status_code=500)

    # 4) Save result
    result = {
        "id": uuid.uuid4().hex,
        "filename": filename,
        "file_url": f"/uploads/{filename}",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "detections": detections,
        "landmarks": landmarks,
        "measurements": measurements
    }
    try:
        save_result(result)
    except Exception as e:
        logger.exception("Failed to save result")
        # still return success payload, but log the persistence error
        return JSONResponse({"status": "warning", "message": f"Result computed but failed to save: {e}", "data": result})

    return JSONResponse({"status": "success", "data": result})


@app.get("/results/")
def get_results():
    return {"status": "success", "data": read_results()}
