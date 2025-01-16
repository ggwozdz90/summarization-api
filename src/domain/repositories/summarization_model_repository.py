from abc import ABC, abstractmethod
from typing import Any, Dict


class SummarizationModelRepository(ABC):
    @abstractmethod
    def summarize(
        self,
        text_to_summarize: str,
        generation_parameters: Dict[str, Any],
    ) -> str:
        pass
