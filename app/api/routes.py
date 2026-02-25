from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image
import numpy as np
import io

from app.services.yolo_service import detect_objects

router = APIRouter()


@router.get("/")
def home():
    return FileResponse("app/templates/index.html")


@router.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except:
        return {"error": "Invalid image"}

    image_np = np.array(image)

    processed_image = detect_objects(image_np)

    image_with_boxes = Image.fromarray(processed_image)

    img_byte_arr = io.BytesIO()
    image_with_boxes.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")