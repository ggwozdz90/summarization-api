from unittest.mock import Mock

import pytest

from core.config.app_config import AppConfig
from core.logger.logger import Logger
from data.factories.summarization_worker_factory import SummarizationWorkerFactory
from data.workers.bart_large_cnn_summarization_worker import (
    BartLargeCnnSummarizationWorker,
)
from domain.exceptions.unsupported_model_configuration_error import (
    UnsupportedModelConfigurationError,
)


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def mock_config() -> AppConfig:
    return Mock(AppConfig)


def test_create_mbart(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given
    mock_config.summarization_model_name = "facebook/bart-large-cnn"
    mock_config.device = "cpu"
    mock_config.summarization_model_download_path = "/path/to/bart"
    mock_config.log_level = "INFO"

    factory = SummarizationWorkerFactory(config=mock_config, logger=mock_logger)

    # When
    worker = factory.create()

    # Then
    assert isinstance(worker, BartLargeCnnSummarizationWorker)
    assert worker._config.device == "cpu"
    assert worker._config.model_name == "facebook/bart-large-cnn"
    assert worker._config.model_download_path == "/path/to/bart"
    assert worker._config.log_level == "INFO"


def test_create_unsupported_model(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given
    mock_config.summarization_model_name = "unsupported-model"
    factory = SummarizationWorkerFactory(config=mock_config, logger=mock_logger)

    # When / Then
    with pytest.raises(UnsupportedModelConfigurationError, match="Unsupported model name: unsupported-model"):
        factory.create()
