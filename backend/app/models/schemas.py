from pydantic import BaseModel

class PredictionResponse(BaseModel):
    status: str
    disease: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
