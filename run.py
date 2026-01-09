"""Server run script."""

import uvicorn
from src.api.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
