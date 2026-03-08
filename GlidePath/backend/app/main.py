# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from starlette.staticfiles import StaticFiles

# from .utils.video import OUTPUT_DIR

# from .routes import analysis, health, weather

# app = FastAPI(title="GlidePath", description="Runway approach assistant")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# # Register routers
# app.include_router(health.router, tags=["health"])
# app.include_router(analysis.router, tags=["analysis"])
# app.include_router(weather.router, tags=["weather"])


# @app.get("/")
# def root():
#     """Root endpoint"""
#     return {"message": "GlidePath backend running"}


# @app.exception_handler(Exception)
# async def unhandled_exception_handler(request: Request, exc: Exception):
#     """Catch-all handler — returns JSON instead of a raw 500 traceback."""
#     return JSONResponse(
#         status_code=500,
#         content={"error": "An unexpected server error occurred."},
#     )


from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from .routes import analysis, health, weather

app = FastAPI(title="GlidePath", description="Runway approach assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use backend/experiments as the output folder for annotated images
BASE_DIR = Path(__file__).resolve().parents[1]   # backend/
OUTPUT_DIR = BASE_DIR / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Register routers
app.include_router(health.router, tags=["health"])
app.include_router(analysis.router, tags=["analysis"])
app.include_router(weather.router, tags=["weather"])


@app.get("/")
def root():
    return {"message": "GlidePath backend running"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected server error occurred."},
    )