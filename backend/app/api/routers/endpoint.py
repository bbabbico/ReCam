from fastapi import FastAPI
from backend.app.api.domain.dto import *

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}

# TODO : 모델 이용 사용자 데이터 입력 API

@app.post("/api/predict")
def predict(req: BuildingInfoRequest):
    return {"name": req.name,
            "category": req.category,
            "area": req.area,
            "peak": req.peak
            }