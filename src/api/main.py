"""
FastAPI main application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import game, shop, combat, optimizer, data

app = FastAPI(
    title="TFT Simulator API",
    description="Teamfight Tactics Set 16 Simulator and Optimization API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(game.router, prefix="/api/game", tags=["Game"])
app.include_router(shop.router, prefix="/api/shop", tags=["Shop"])
app.include_router(combat.router, prefix="/api/combat", tags=["Combat"])
app.include_router(optimizer.router, prefix="/api/optimizer", tags=["Optimizer"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])


@app.get("/")
async def root():
    """API status check."""
    return {
        "status": "ok",
        "name": "TFT Simulator API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
