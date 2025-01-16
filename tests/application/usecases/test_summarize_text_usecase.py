from unittest.mock import Mock

import pytest

from application.usecases.summarize_text_usecase import SummarizeTextUseCase
from core.config.app_config import AppConfig
from core.logger.logger import Logger
from domain.services.summarization_service import SummarizationService


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def mock_config() -> AppConfig:
    return Mock(AppConfig)


@pytest.fixture
def mock_summarization_service() -> SummarizationService:
    return Mock(SummarizationService)


@pytest.fixture
def use_case(
    mock_config: AppConfig,
    mock_logger: Logger,
    mock_summarization_service: SummarizationService,
) -> SummarizeTextUseCase:
    return SummarizeTextUseCase(
        config=mock_config,
        logger=mock_logger,
        summarization_service=mock_summarization_service,
    )


@pytest.mark.asyncio
async def test_execute_success(
    use_case: SummarizeTextUseCase,
    mock_summarization_service: Mock,
) -> None:
    # Given
    mock_summarization_service.summarize_text = Mock(return_value="result")

    # When
    result = await use_case.execute("Hello", {})

    # Then
    assert result == "result"
    mock_summarization_service.summarize_text.assert_called_once_with("Hello", {})
