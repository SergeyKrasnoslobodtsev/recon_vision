from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes.api import router as api_router
from app.api.routes.docs import create_docs_router
from app.bootstrap.lifespan import create_lifespan
from app.infrastructure.config.settings import get_settings
from app.infrastructure.logging.setup import configure_logging


settings = get_settings()
configure_logging(settings.logging_level)


def get_application() -> FastAPI:
    """Создаёт и настраивает экземпляр FastAPI приложения.

    Returns:
        FastAPI: Сконфигурированное приложение.
    """
    application = FastAPI(
        title=settings.project_name,
        debug=settings.debug,
        version=settings.version,
        docs_url=None,
        redoc_url=None,
        lifespan=create_lifespan(settings),
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        application.mount(
            "/static",
            StaticFiles(directory=settings.docs_dir),
            name="static",
        )
    except Exception:
        pass

    application.include_router(api_router, prefix=settings.api_prefix)
    application.include_router(
        create_docs_router(application),
        prefix=f"{settings.api_prefix}/v1",
    )

    return application


app = get_application()
