from fastapi import APIRouter

from app.api.routes import route


router = APIRouter()

router.include_router(route.router, prefix="/v1")
