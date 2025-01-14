from unittest.mock import Mock, patch

import pytest

from core.config.app_config import AppConfig
from core.logger.logger import Logger
from core.timer.timer import Timer, TimerFactory
from data.factories.summarization_worker_factory import SummarizationWorkerFactory
from data.repositories.summarization_model_repository_impl import (
    SummarizationModelRepositoryImpl,
)
from domain.repositories.directory_repository import DirectoryRepository


@pytest.fixture
def mock_config() -> AppConfig:
    config = Mock(AppConfig)
    config.summarization_model_download_path = "/models"
    config.device = "cpu"
    config.summarization_model_name = "facebook/bart"
    config.model_idle_timeout = 60
    return config


@pytest.fixture
def mock_directory_repository() -> Mock:
    return Mock(spec=DirectoryRepository)


@pytest.fixture
def mock_timer() -> Mock:
    return Mock(spec=Timer)


@pytest.fixture
def mock_timer_factory(mock_timer: Mock) -> Mock:
    factory = Mock(spec=TimerFactory)
    factory.create.return_value = mock_timer
    return factory


@pytest.fixture
def mock_logger() -> Mock:
    return Mock(spec=Logger)


@pytest.fixture
def mock_worker() -> Mock:
    return Mock()


@pytest.fixture
def mock_worker_factory(mock_worker: Mock) -> Mock:
    factory = Mock(spec=SummarizationWorkerFactory)
    factory.create.return_value = mock_worker
    return factory


@pytest.fixture
def summarize_model_repository_impl(
    mock_config: Mock,
    mock_directory_repository: Mock,
    mock_timer_factory: Mock,
    mock_logger: Mock,
    mock_worker_factory: Mock,
) -> SummarizationModelRepositoryImpl:
    with patch.object(SummarizationModelRepositoryImpl, "_instance", None):
        return SummarizationModelRepositoryImpl(
            config=mock_config,
            directory_repository=mock_directory_repository,
            timer_factory=mock_timer_factory,
            logger=mock_logger,
            worker_factory=mock_worker_factory,
        )


def test_summarize_success(
    summarize_model_repository_impl: SummarizationModelRepositoryImpl,
    mock_worker: Mock,
    mock_timer: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = False
    mock_worker.summarize.return_value = "text"

    # When
    result = summarize_model_repository_impl.summarize("text to summarize")

    # Then
    assert result == "text"
    mock_worker.start.assert_called_once()
    mock_worker.summarize.assert_called_once_with("text to summarize")
    mock_timer.start.assert_called_once_with(60, summarize_model_repository_impl._check_idle_timeout)


def test_check_idle_timeout_stops_worker(
    summarize_model_repository_impl: SummarizationModelRepositoryImpl,
    mock_worker: Mock,
    mock_timer: Mock,
    mock_logger: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = True
    mock_worker.is_processing.return_value = False

    # When
    summarize_model_repository_impl._check_idle_timeout()

    # Then
    mock_worker.stop.assert_called_once()
    mock_timer.cancel.assert_called_once()
    mock_logger.debug.assert_any_call("Checking summarization model idle timeout")
    mock_logger.info.assert_any_call("Summarization model stopped due to idle timeout")
