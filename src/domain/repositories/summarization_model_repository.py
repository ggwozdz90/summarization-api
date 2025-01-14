from abc import ABC, abstractmethod


class SummarizationModelRepository(ABC):
    @abstractmethod
    def summarize(
        self,
        text: str,
    ) -> str:
        pass
