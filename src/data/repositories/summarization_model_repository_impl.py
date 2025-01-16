import threading
import time
from typing import Annotated, Any, Dict, Optional

from fastapi import Depends

from core.config.app_config import AppConfig
from core.logger.logger import Logger
from core.timer.timer import TimerFactory
from data.factories.summarization_worker_factory import SummarizationWorkerFactory
from data.repositories.directory_repository_impl import DirectoryRepositoryImpl
from domain.repositories.directory_repository import DirectoryRepository
from domain.repositories.summarization_model_repository import (
    SummarizationModelRepository,
)


class SummarizationModelRepositoryImpl(SummarizationModelRepository):  # type: ignore
    _instance: Optional["SummarizationModelRepositoryImpl"] = None
    _lock = threading.Lock()

    def __new__(
        cls,
        config: Annotated[AppConfig, Depends()],
        directory_repository: Annotated[DirectoryRepository, Depends(DirectoryRepositoryImpl)],
        timer_factory: Annotated[TimerFactory, Depends()],
        logger: Annotated[Logger, Depends()],
        worker_factory: Annotated[SummarizationWorkerFactory, Depends()],
    ) -> "SummarizationModelRepositoryImpl":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SummarizationModelRepositoryImpl, cls).__new__(cls)
                    cls._instance._initialize(config, directory_repository, timer_factory, logger, worker_factory)

        return cls._instance

    def _initialize(
        self,
        config: AppConfig,
        directory_repository: DirectoryRepository,
        timer_factory: TimerFactory,
        logger: Logger,
        worker_factory: SummarizationWorkerFactory,
    ) -> None:
        directory_repository.create_directory(config.summarization_model_download_path)
        self.config = config
        self.timer = timer_factory.create()
        self.logger = logger
        self.worker = worker_factory.create()
        self.last_access_time = 0.0

    def _check_idle_timeout(self) -> None:
        self.logger.debug("Checking summarization model idle timeout")

        if self.worker.is_alive() and not self.worker.is_processing():
            with self._lock:
                self.worker.stop()
                self.timer.cancel()
                self.logger.info("Summarization model stopped due to idle timeout")

    def summarize(
        self,
        text_to_summarize: str,
        generation_parameters: Dict[str, Any],
    ) -> str:
        with self._lock:
            if not self.worker.is_alive():
                self.logger.info("Starting summarization worker")
                self.worker.start()

        self.logger.debug("Summarization started")

        result: str = self.worker.summarize(
            text_to_summarize,
            generation_parameters,
        )

        self.timer.start(
            self.config.model_idle_timeout,
            self._check_idle_timeout,
        )

        self.last_access_time = time.time()

        self.logger.debug("Summarization completed")

        return result
