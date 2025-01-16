from typing import Annotated, Any, Dict

from fastapi import Depends

from core.config.app_config import AppConfig
from core.logger.logger import Logger
from data.repositories.summarization_model_repository_impl import (
    SummarizationModelRepositoryImpl,
)
from domain.repositories.summarization_model_repository import (
    SummarizationModelRepository,
)


class SummarizationService:
    def __init__(
        self,
        config: Annotated[AppConfig, Depends()],
        summarization_model_repository: Annotated[
            SummarizationModelRepository,
            Depends(SummarizationModelRepositoryImpl),
        ],
        logger: Annotated[Logger, Depends()],
    ) -> None:
        self.config = config
        self.summarization_model_repository = summarization_model_repository
        self.logger = logger

    def summarize_text(
        self,
        text_to_summarize: str,
        generation_parameters: Dict[str, Any],
    ) -> str:
        self.logger.debug("Starting summarization of text")

        summary: str = self.summarization_model_repository.summarize(
            text_to_summarize,
            generation_parameters,
        )

        self.logger.debug("Completed summarization of text")

        return summary
