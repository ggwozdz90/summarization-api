from typing import Annotated

from fastapi import Depends

from core.config.app_config import AppConfig
from core.logger.logger import Logger
from domain.services.summarization_service import SummarizationService


class SummarizeTextUseCase:
    def __init__(
        self,
        config: Annotated[AppConfig, Depends()],
        logger: Annotated[Logger, Depends()],
        summarization_service: Annotated[SummarizationService, Depends()],
    ) -> None:
        self.config = config
        self.logger = logger
        self.summarization_service = summarization_service

    async def execute(
        self,
        text: str,
    ) -> str:
        self.logger.info(f"Executing summarization for text '{text}'")

        summarization_result: str = self.summarization_service.summarize_text(text)

        self.logger.info("Returning summarization result")

        return summarization_result
