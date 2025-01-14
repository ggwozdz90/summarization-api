from typing import Annotated

from fastapi import Depends

from core.config.app_config import AppConfig
from core.logger.logger import Logger
from data.workers.bart_large_cnn_summarization_worker import (
    BartLargeCnnSummarizationConfig,
    BartLargeCnnSummarizationWorker,
)
from domain.exceptions.unsupported_model_configuration_error import (
    UnsupportedModelConfigurationError,
)


class SummarizationWorkerFactory:
    def __init__(
        self,
        config: Annotated[AppConfig, Depends()],
        logger: Annotated[Logger, Depends()],
    ):
        self.config = config
        self.logger = logger

    def create(self) -> BartLargeCnnSummarizationWorker:
        if self.config.summarization_model_name == "facebook/bart-large-cnn":
            return BartLargeCnnSummarizationWorker(
                BartLargeCnnSummarizationConfig(
                    device=self.config.device,
                    model_name=self.config.summarization_model_name,
                    model_download_path=self.config.summarization_model_download_path,
                    log_level=self.config.log_level,
                ),
                logger=self.logger,
            )
        else:
            raise UnsupportedModelConfigurationError(self.config.summarization_model_name)
