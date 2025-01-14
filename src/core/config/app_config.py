import os
from typing import Optional

from dotenv import load_dotenv

from core.logger.logger import Logger


class AppConfig:
    _instance: Optional["AppConfig"] = None

    log_level: Optional[str]
    device: Optional[str]
    fastapi_host: Optional[str]
    fastapi_port: Optional[int]
    summarization_model_name: Optional[str]
    summarization_model_download_path: Optional[str]
    model_idle_timeout: Optional[int]

    def __new__(cls) -> "AppConfig":
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)

        return cls._instance

    def _load_env_variables(self) -> None:
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.device = os.getenv("DEVICE", "cpu")
        self.fastapi_host = os.getenv("FASTAPI_HOST", "127.0.0.1")
        self.model_idle_timeout = int(os.getenv("MODEL_IDLE_TIMEOUT", "60"))
        self.summarization_model_name = os.getenv("SUMMARIZATION_MODEL_NAME", "facebook/bart-large-cnn")
        self.summarization_model_download_path = os.getenv(
            "SUMMARIZATION_MODEL_DOWNLOAD_PATH",
            "downloaded_summarization_models",
        )
        try:
            self.fastapi_port = int(os.getenv("FASTAPI_PORT", "8000"))
        except ValueError:
            self.fastapi_port = 8000

    def initialize(
        self,
        logger: Logger,
    ) -> None:
        logger.info("Initializing configuration...")
        load_dotenv()
        self._load_env_variables()
        config_message = (
            f"Configuration loaded:\n"
            f"LOG_LEVEL: {self.log_level}\n"
            f"DEVICE: {self.device}\n"
            f"FASTAPI_HOST: {self.fastapi_host}\n"
            f"FASTAPI_PORT: {self.fastapi_port}\n"
            f"SUMMARIZATION_MODEL_NAME: {self.summarization_model_name}\n"
            f"SUMMARIZATION_MODEL_DOWNLOAD_PATH: {self.summarization_model_download_path}\n"
            f"MODEL_IDLE_TIMEOUT: {self.model_idle_timeout}"
        )
        logger.info(config_message)
        logger.info("Configuration initialized successfully.")
