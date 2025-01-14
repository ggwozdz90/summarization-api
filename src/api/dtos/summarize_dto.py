from pydantic import BaseModel


class SummarizeDTO(BaseModel):
    text: str
