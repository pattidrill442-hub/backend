from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Breast Cancer Detection API")

# Allow mobile app/frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "final_mobilenetv2_breast_cancer_3class.keras"
IMG_SIZE = (224, 224)

MIN_WIDTH = 100
MIN_HEIGHT = 100

# Thresholds
CONFIDENCE_THRESHOLD = 0.70       # for benign/malignant acceptance
INVALID_THRESHOLD = 0.70          # only reject as invalid if invalid score is really high

# IMPORTANT:
# This order should match the class_names from your training dataset.
# With folders benign / invalid / malignant loaded alphabetically,
# TensorFlow will usually use this exact order:
CLASS_NAMES = ["benign", "invalid", "malignant"]

BENIGN_INDEX = CLASS_NAMES.index("benign")
INVALID_INDEX = CLASS_NAMES.index("invalid")
MALIGNANT_INDEX = CLASS_NAMES.index("malignant")

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)


def validate_uploaded_image(image_bytes: bytes, content_type: str | None):
    if not content_type or not content_type.startswith("image/"):
        return False, "Unsupported file type", "Please upload a valid image file."

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return False, "Invalid image", "Could not process the uploaded file as an image."
    except Exception:
        return False, "Invalid image", "Could not process the uploaded image."

    width, height = image.size
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False, "Image too small", "Please upload a clearer and higher-quality image."

    return True, None, None


def prepare_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array


@app.get("/")
def root():
    return {"message": "Breast Cancer Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # Step 1: basic file validation
        is_valid, error_title, error_message = validate_uploaded_image(
            image_bytes=image_bytes,
            content_type=file.content_type,
        )

        if not is_valid:
            return {
                "success": False,
                "status": "invalid",
                "error": error_title,
                "message": error_message,
            }

        # Step 2: preprocess
        processed_image = prepare_image(image_bytes)

        # Step 3: predict 3-class probabilities
        predictions = model.predict(processed_image, verbose=0)[0]

        benign_score = float(predictions[BENIGN_INDEX])
        invalid_score = float(predictions[INVALID_INDEX])
        malignant_score = float(predictions[MALIGNANT_INDEX])

        raw_scores = {
            "benign": round(benign_score, 4),
            "invalid": round(invalid_score, 4),
            "malignant": round(malignant_score, 4),
        }

        # Step 4: reject as invalid only if the invalid score is strongly high
        if invalid_score >= INVALID_THRESHOLD:
            return {
                "success": False,
                "status": "invalid",
                "filename": file.filename,
                "error": "Unsupported image",
                "message": "This does not appear to be a valid breast histopathology image. Please upload a proper tissue slide image.",
                "confidence": round(invalid_score, 4),
                "raw_scores": raw_scores,
                "note": "For educational and research purposes only. Not a medical diagnosis.",
            }

        # Step 5: otherwise decide only between benign and malignant
        if benign_score >= malignant_score:
            predicted_label = "benign"
            confidence = benign_score
        else:
            predicted_label = "malignant"
            confidence = malignant_score

        # Step 6: uncertainty handling for benign/malignant
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "success": False,
                "status": "uncertain",
                "filename": file.filename,
                "error": "Uncertain prediction",
                "message": "The model is not confident enough in this result. Please upload a clearer image or seek manual review.",
                "confidence": round(confidence, 4),
                "raw_scores": raw_scores,
                "note": "For educational and research purposes only. Not a medical diagnosis.",
            }

        # Step 7: confident benign/malignant result
        return {
            "success": True,
            "status": "ok",
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "raw_scores": raw_scores,
            "note": "For educational and research purposes only. Not a medical diagnosis.",
        }

    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": "Server error",
            "message": str(e),
        }
