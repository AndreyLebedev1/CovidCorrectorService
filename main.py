from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from model import ModelCorrected


app = FastAPI()
model = ModelCorrected()


class PredictRequest(BaseModel):
    path: str


@app.post("/predict")
def predict(request: PredictRequest):
    image_path = Path(request.path)

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    try:
        with Image.open(image_path) as image:
            return model.inference(image)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
