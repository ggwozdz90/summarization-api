from typing import Annotated, Any, Dict

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
        text_to_summarize: str,
        generation_parameters: Dict[str, Any],
    ) -> str:
        self.logger.info(f"Executing summarization for text '{text_to_summarize}'")

        summary: str = self.summarization_service.summarize_text(
            text_to_summarize,
            generation_parameters,
        )

        self.logger.info("Returning summarization result")

        return summary
