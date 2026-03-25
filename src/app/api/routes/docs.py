"""Определяет маршруты статической документации API."""

from fastapi import APIRouter, FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html


def create_docs_router(app: FastAPI) -> APIRouter:
    """Создаёт роутер для статической документации.

    Args:
        app: Экземпляр FastAPI приложения.

    Returns:
        APIRouter: Роутер со swagger и redoc.
    """

    router = APIRouter()

    @router.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @router.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " - ReDoc",
            redoc_js_url="/static/redoc.standalone.js",
        )

    return router
