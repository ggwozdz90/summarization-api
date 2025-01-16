import multiprocessing
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest

from core.logger.logger import Logger
from data.workers.bart_large_cnn_summarization_worker import (
    BartLargeCnnSummarizationConfig,
    BartLargeCnnSummarizationWorker,
)


@pytest.fixture
def bart_config() -> BartLargeCnnSummarizationConfig:
    return BartLargeCnnSummarizationConfig(
        device="cuda",
        model_name="facebook/mbart-large-50",
        model_download_path="/tmp",
        log_level="INFO",
    )


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def bart_worker(
    bart_config: BartLargeCnnSummarizationConfig,
    mock_logger: Logger,
) -> Generator[BartLargeCnnSummarizationWorker, None, None]:
    worker = BartLargeCnnSummarizationWorker(bart_config, mock_logger)
    yield worker
    worker.stop()


class MockTensor:
    def to(self, device: str) -> "MockTensor":
        return self


def test_summarize_sends_correct_command(bart_worker: BartLargeCnnSummarizationWorker) -> None:
    with (
        patch("multiprocessing.Process") as MockProcess,
        patch(
            "data.workers.bart_large_cnn_summarization_worker.BartForConditionalGeneration.from_pretrained",
        ) as mock_load_model,
        patch(
            "data.workers.bart_large_cnn_summarization_worker.AutoTokenizer.from_pretrained",
        ) as mock_load_tokenizer,
    ):
        mock_process = Mock()
        MockProcess.return_value = mock_process
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = mock_model
        mock_load_tokenizer.return_value = mock_tokenizer

        # Given
        bart_worker.start()
        text_to_summarize = "Hello, world!"

        # When
        with (
            patch.object(bart_worker._pipe_parent, "send") as mock_send,
            patch.object(
                bart_worker._pipe_parent,
                "recv",
                return_value="Hello",
            ),
        ):
            bart_worker.summarize(text_to_summarize, {})

            # Then
            mock_send.assert_called_once_with(("summarize", (text_to_summarize, {})))


def test_summarize_raises_error_if_worker_not_running(bart_worker: BartLargeCnnSummarizationWorker) -> None:
    # Given
    text_to_summarize = "Hello, world!"

    # When / Then
    with pytest.raises(RuntimeError, match="Worker process is not running"):
        bart_worker.summarize(text_to_summarize, {})


def test_initialize_shared_object(bart_config: BartLargeCnnSummarizationConfig, mock_logger: Logger) -> None:
    worker = BartLargeCnnSummarizationWorker(bart_config, mock_logger)
    with (
        patch(
            "data.workers.bart_large_cnn_summarization_worker.BartForConditionalGeneration.from_pretrained",
        ) as mock_load_model,
        patch(
            "data.workers.bart_large_cnn_summarization_worker.AutoTokenizer.from_pretrained",
        ) as mock_load_tokenizer,
    ):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_load_tokenizer.return_value = mock_tokenizer

        # When
        model, tokenizer = worker.initialize_shared_object(bart_config)

        # Then
        mock_load_model.assert_called_once_with(
            bart_config.model_name,
            cache_dir=bart_config.model_download_path,
        )
        mock_load_tokenizer.assert_called_once_with(
            bart_config.model_name,
            cache_dir=bart_config.model_download_path,
        )
        assert model == mock_model
        assert tokenizer == mock_tokenizer


def test_handle_command_summarize(
    bart_worker: BartLargeCnnSummarizationWorker,
    bart_config: BartLargeCnnSummarizationConfig,
) -> None:
    with (
        patch(
            "data.workers.bart_large_cnn_summarization_worker.BartForConditionalGeneration.from_pretrained",
        ) as mock_load_model,
        patch(
            "data.workers.bart_large_cnn_summarization_worker.AutoTokenizer.from_pretrained",
        ) as mock_load_tokenizer,
        patch("torch.no_grad"),
    ):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = mock_model
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_is_processing = multiprocessing.Value("b", False)
        mock_processing_lock = multiprocessing.Lock()
        pipe = Mock()

        # When
        bart_worker.handle_command(
            command="summarize",
            args=("Hello, world!", {}),
            shared_object=(mock_model, mock_tokenizer),
            config=bart_config,
            pipe=pipe,
            is_processing=mock_is_processing,
            processing_lock=mock_processing_lock,
        )

        # Then
        assert not mock_is_processing.value
        mock_tokenizer.assert_called_once_with(
            ["Hello, world!"],
            max_length=1024,
            return_tensors="pt",
        )


def test_handle_command_summarize_error(
    bart_worker: BartLargeCnnSummarizationWorker,
    bart_config: BartLargeCnnSummarizationConfig,
) -> None:
    with (
        patch(
            "data.workers.bart_large_cnn_summarization_worker.BartForConditionalGeneration.from_pretrained",
        ) as mock_load_model,
        patch(
            "data.workers.bart_large_cnn_summarization_worker.AutoTokenizer.from_pretrained",
        ) as mock_load_tokenizer,
        patch("torch.no_grad"),
    ):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = mock_model
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_is_processing = multiprocessing.Value("b", False)
        mock_processing_lock = multiprocessing.Lock()
        pipe = Mock()
        mock_model.generate.side_effect = RuntimeError("Summarize error")

        # When
        bart_worker.handle_command(
            command="summarize",
            args=("Hello, world!", {}),
            shared_object=(mock_model, mock_tokenizer),
            config=bart_config,
            pipe=pipe,
            is_processing=mock_is_processing,
            processing_lock=mock_processing_lock,
        )

        # Then
        assert not mock_is_processing.value
        assert pipe.send.call_count == 1
        assert isinstance(pipe.send.call_args[0][0], RuntimeError)
        assert pipe.send.call_args[0][0].args[0] == "Summarize error"
