from typing import Annotated

from fastapi import APIRouter, Body, Depends

from api.dtos.summarize_dto import SummarizeDTO
from api.dtos.summarize_result_dto import SummarizeResultDTO
from application.usecases.summarize_text_usecase import SummarizeTextUseCase


class SummarizeRouter:
    def __init__(self) -> None:
        self.router = APIRouter()
        self.router.post("/summarize")(self.summarize)

    async def summarize(
        self,
        summarize_text_usecase: Annotated[SummarizeTextUseCase, Depends()],
        summarize_dto: SummarizeDTO = Body(...),
    ) -> SummarizeResultDTO:
        summary = await summarize_text_usecase.execute(
            summarize_dto.text_to_summarize,
            summarize_dto.generation_parameters,
        )

        return SummarizeResultDTO(summary=summary)
