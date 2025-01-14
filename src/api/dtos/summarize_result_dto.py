from pydantic import BaseModel


class SummarizeResultDTO(BaseModel):
    content: str
