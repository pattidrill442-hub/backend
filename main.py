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
    allow_origins=["*"],  # later you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "final_mobilenetv2_breast_cancer.keras"
IMG_SIZE = (224, 224)

# Basic validation settings
MIN_WIDTH = 100
MIN_HEIGHT = 100
UNCERTAINTY_THRESHOLD = 0.75

# Load model once when API starts
model = tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array


def validate_uploaded_image(
    image_bytes: bytes,
    content_type: str | None,
) -> tuple[bool, str | None, str | None]:
    # Check declared content type first
    if not content_type or not content_type.startswith("image/"):
        return (
            False,
            "Unsupported file type",
            "Please upload a valid image file.",
        )

    # Check actual image validity
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return (
            False,
            "Invalid image",
            "Could not process the uploaded file as an image.",
        )
    except Exception:
        return (
            False,
            "Invalid image",
            "Could not process the uploaded image.",
        )

    width, height = image.size
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return (
            False,
            "Image too small",
            "Please upload a clearer and higher-quality image.",
        )

    return True, None, None


@app.get("/")
def root():
    return {
        "message": "Breast Cancer Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # Step 1: Validate uploaded image
        is_valid, error_title, error_message = validate_uploaded_image(
            image_bytes=image_bytes,
            content_type=file.content_type,
        )

        if not is_valid:
            return {
                "success": False,
                "error": error_title,
                "message": error_message,
            }

        # Step 2: Prepare image for model
        processed_image = prepare_image(image_bytes)

        # Step 3: Run prediction
        prediction = float(model.predict(processed_image, verbose=0)[0][0])

        # Step 4: Reject uncertain / likely unsupported inputs
        confidence_score = max(prediction, 1 - prediction)

        if confidence_score < UNCERTAINTY_THRESHOLD:
            return {
                "success": False,
                "error": "Uncertain prediction",
                "message": (
                    "This image may not be a valid breast histopathology image. "
                    "Please upload a clearer or relevant histopathology image."
                ),
            }

        # Step 5: Final classification
        if prediction > 0.5:
            label = "malignant"
            confidence = prediction
        else:
            label = "benign"
            confidence = 1 - prediction

        return {
            "success": True,
            "filename": file.filename,
            "prediction": label,
            "confidence": round(float(confidence), 4),
            "raw_score": round(prediction, 4),
            "note": "For educational and research purposes only. Not a medical diagnosis."
        }

    except Exception as e:
        return {
            "success": False,
            "error": "Server error",
            "message": str(e),
        }
