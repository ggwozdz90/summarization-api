import multiprocessing
import multiprocessing.connection
import multiprocessing.synchronize
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Dict, Tuple

from transformers import AutoTokenizer, BartForConditionalGeneration

from data.workers.base_worker import BaseWorker
from domain.exceptions.worker_not_running_error import WorkerNotRunningError


@dataclass
class BartLargeCnnSummarizationConfig:
    device: str
    model_name: str
    model_download_path: str
    log_level: str


class BartLargeCnnSummarizationWorker(
    BaseWorker[  # type: ignore
        Tuple[str, Dict[str, Any]],
        str,
        BartLargeCnnSummarizationConfig,
        Tuple[BartForConditionalGeneration, AutoTokenizer],
    ],
):
    def summarize(
        self,
        text_to_summarize: str,
        generation_parameters: Dict[str, Any],
    ) -> str:
        if not self.is_alive():
            raise WorkerNotRunningError()

        self._pipe_parent.send(
            (
                "summarize",
                (
                    text_to_summarize,
                    generation_parameters,
                ),
            ),
        )
        result = self._pipe_parent.recv()

        if isinstance(result, Exception):
            raise result

        return str(result)

    def initialize_shared_object(
        self,
        config: BartLargeCnnSummarizationConfig,
    ) -> Tuple[BartForConditionalGeneration, AutoTokenizer]:
        model = BartForConditionalGeneration.from_pretrained(
            config.model_name,
            cache_dir=config.model_download_path,
        ).to(config.device)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.model_download_path,
        )
        return model, tokenizer

    def handle_command(
        self,
        command: str,
        args: Tuple[str, Dict[str, Any]],
        shared_object: Tuple[BartForConditionalGeneration, AutoTokenizer],
        config: BartLargeCnnSummarizationConfig,
        pipe: multiprocessing.connection.Connection,
        is_processing: Synchronized,  # type: ignore
        processing_lock: multiprocessing.synchronize.Lock,
    ) -> None:
        if command == "summarize":
            try:
                with processing_lock:
                    is_processing.value = True

                text_to_summarize, generation_parameters = args
                model, tokenizer = shared_object

                inputs = tokenizer([text_to_summarize], max_length=1024, return_tensors="pt").to(config.device)

                summary_ids = model.generate(inputs["input_ids"], **generation_parameters)

                output = tokenizer.batch_decode(
                    summary_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                pipe.send("".join(output))

            except Exception as e:
                pipe.send(e)

            finally:
                with processing_lock:
                    is_processing.value = False

    def get_worker_name(self) -> str:
        return type(self).__name__
