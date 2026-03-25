"""Определяет служебные маршруты healthcheck."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """Возвращает статус доступности сервиса.

    Returns:
        dict[str, str]: Признак доступности приложения.
    """
    return {"status": "ok"}
