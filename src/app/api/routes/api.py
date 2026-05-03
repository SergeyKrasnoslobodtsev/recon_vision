from fastapi import APIRouter

from app.api.routes import health, reconciliation

router = APIRouter()

router.include_router(health.router)
router.include_router(reconciliation.router)
