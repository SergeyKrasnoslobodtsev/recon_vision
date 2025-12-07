from api.routes.api import router as api_router
from core.config import (
    API_PREFIX,
    DEBUG,
    PROJECT_NAME,
    VERSION,
    DOCS_DIR,
    TTL_CACHE_IN_HOURS,
)
from core.events import create_start_app_handler
from api.routes.static_docs import create_docs_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
from app.services.cache_in_disk import CacheInMemory


@asynccontextmanager
async def lifespan(_: FastAPI):
    cache = CacheInMemory(expire=TTL_CACHE_IN_HOURS * 3600)
    cleanup_task = asyncio.create_task(cache.cleanup_expired_files())
    yield

    cleanup_task.cancel()  # остановка фоновой задачи очистки при завершении приложения
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


def get_application() -> FastAPI:
    application = FastAPI(
        title=PROJECT_NAME,
        debug=DEBUG,
        version=VERSION,
        docs_url=None,
        redoc_url=None,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        application.mount("/static", StaticFiles(directory=DOCS_DIR), name="static")
    except Exception:
        pass

    application.include_router(api_router, prefix=API_PREFIX)
    application.include_router(
        create_docs_router(application), prefix=f"{API_PREFIX}/v1"
    )
    application.add_event_handler("startup", create_start_app_handler(application))

    return application


app = get_application()
