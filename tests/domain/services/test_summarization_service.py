from unittest.mock import Mock

import pytest

from core.config.app_config import AppConfig
from core.logger.logger import Logger
from domain.repositories.summarization_model_repository import (
    SummarizationModelRepository,
)
from domain.services.summarization_service import SummarizationService


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def mock_config() -> AppConfig:
    config = Mock(AppConfig)
    config.summarization_model_name = "test_model"
    return config


@pytest.fixture
def mock_summarization_model_repository() -> SummarizationModelRepository:
    return Mock(SummarizationModelRepository)


@pytest.fixture
def summarization_service(
    mock_logger: Logger,
    mock_config: AppConfig,
    mock_summarization_model_repository: SummarizationModelRepository,
) -> SummarizationService:
    return SummarizationService(
        logger=mock_logger,
        config=mock_config,
        summarization_model_repository=mock_summarization_model_repository,
    )


def test_summarize_text_success(
    summarization_service: SummarizationService,
    mock_summarization_model_repository: SummarizationModelRepository,
) -> None:
    # Given
    text_to_summarize = "Hello World"
    mock_summarization_model_repository.summarize.return_value = "Hello"

    # When
    result = summarization_service.summarize_text(text_to_summarize, {})

    # Then
    assert result == "Hello"
    mock_summarization_model_repository.summarize.assert_called_once_with(text_to_summarize, {})


def test_summarize_text_exception(
    summarization_service: SummarizationService,
    mock_summarization_model_repository: SummarizationModelRepository,
) -> None:
    # Given
    text_to_summarize = "Hello World"
    mock_summarization_model_repository.summarize.side_effect = Exception("Summarization error")

    # When / Then
    with pytest.raises(Exception, match="Summarization error"):
        summarization_service.summarize_text(text_to_summarize, {})
