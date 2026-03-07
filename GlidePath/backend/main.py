# Minimal FastAPI backend for GlidePath
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Root endpoint
@app.get("/")
def root():
    """Root endpoint for backend status"""
    return {"message": "GlidePath backend running"}

# Health check endpoint
@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"}

# Analyze sample endpoint (mock)
@app.get("/analyze-sample")
def analyze_sample():
    """Mock runway analysis results"""
    return {
        "alignment": "centered",
        "stability": "stable",
        "confidence": 0.92,
        "wind_direction": "NE",
        "wind_speed": 7.5
    }