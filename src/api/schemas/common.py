"""
Common API schemas.
"""

from pydantic import BaseModel
from typing import Optional, List, Any
from enum import Enum


class ResponseStatus(str, Enum):
    """Response status enum."""

    SUCCESS = "success"
    ERROR = "error"


class BaseResponse(BaseModel):
    """Base response model."""

    status: ResponseStatus = ResponseStatus.SUCCESS
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""

    status: ResponseStatus = ResponseStatus.ERROR
    error: str
    detail: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Paginated response model."""

    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
