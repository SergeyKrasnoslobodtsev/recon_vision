from fastapi import APIRouter

from app.api.routes import health, reconciliation


router = APIRouter()

router.include_router(health.router, prefix="/v1")
router.include_router(reconciliation.router, prefix="/v1")
