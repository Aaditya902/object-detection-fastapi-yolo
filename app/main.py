from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="AI Vision Detection Service",
    description="YOLOv8-based handbag component detection",
    version="1.0"
)

app.include_router(router)