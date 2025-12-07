from typing import Callable

from fastapi import FastAPI
from loguru import logger


def create_start_app_handler(app: FastAPI) -> Callable:
    async def start_app() -> None:
        logger.info(f"Starting application: {app.title} (version: {app.version})")
        # Additional startup tasks can be added here

    return start_app
