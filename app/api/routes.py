from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/api")


class TagsRequest(BaseModel):
    tags: list[str]


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "iastech-api",
    }

