from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Breast Cancer Detection API")

# Allow mobile app/frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "final_mobilenetv2_breast_cancer.keras"
IMG_SIZE = (224, 224)

# Load model once when API starts
model = tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array


@app.get("/")
def root():
    return {
        "message": "Breast Cancer Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_image = prepare_image(image_bytes)

        prediction = model.predict(processed_image, verbose=0)[0][0]

        if prediction > 0.5:
            label = "malignant"
            confidence = float(prediction)
        else:
            label = "benign"
            confidence = float(1 - prediction)

        return {
            "success": True,
            "filename": file.filename,
            "prediction": label,
            "confidence": round(confidence, 4),
            "raw_score": round(float(prediction), 4),
            "note": "For educational and research purposes only. Not a medical diagnosis."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }