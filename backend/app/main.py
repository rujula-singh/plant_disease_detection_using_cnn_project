from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .services.prediction import predict_image

app = FastAPI(title="Plant Disease Detection API")

# Allow CORS for communicating with a decoupled frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this explicitly in production (e.g. 'http://localhost:5173')
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Plant Disease Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate the file uploaded is indeed an image format
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    image_bytes = await file.read()
    
    try:
        prediction, confidence = predict_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
    return {
        "status": "success",
        "disease": prediction,
        "confidence": round(float(confidence), 4)
    }

@app.post("/predict/bulk")
async def predict_bulk(files: List[UploadFile] = File(...)):
    """
    Accepts multiple image files and returns predictions for all of them
    in a single request. Individual image failures do NOT cause the entire
    request to fail — each result clearly indicates success or failure.
    """
    results = []
    success_count = 0
    failed_count = 0

    for file in files:
        # Validate image MIME type
        if not file.content_type.startswith("image/"):
            results.append({
                "image": file.filename,
                "status": "failed",
                "error": "Invalid image format",
            })
            failed_count += 1
            continue

        try:
            image_bytes = await file.read()
            prediction, confidence = predict_image(image_bytes)
            results.append({
                "image": file.filename,
                "status": "success",
                "prediction": prediction,
                "confidence": round(float(confidence), 4),
            })
            success_count += 1
        except Exception as e:
            results.append({
                "image": file.filename,
                "status": "failed",
                "error": str(e),
            })
            failed_count += 1

    return {
        "total": len(files),
        "success": success_count,
        "failed": failed_count,
        "results": results,
    }
