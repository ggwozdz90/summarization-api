from pydantic import BaseModel


class SummarizeResultDTO(BaseModel):
    summary: str
