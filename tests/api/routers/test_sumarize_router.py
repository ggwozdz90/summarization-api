from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.summarize_router import SummarizeRouter
from application.usecases.summarize_text_usecase import SummarizeTextUseCase


@pytest.fixture
def mock_summarize_text_usecase() -> SummarizeTextUseCase:
    return Mock(SummarizeTextUseCase)


@pytest.fixture
def client(
    mock_summarize_text_usecase: SummarizeTextUseCase,
) -> TestClient:
    router = SummarizeRouter()
    app = FastAPI()
    app.include_router(router.router)
    app.dependency_overrides[SummarizeTextUseCase] = lambda: mock_summarize_text_usecase
    return TestClient(app)


def test_summarize_success(
    client: TestClient,
    mock_summarize_text_usecase: SummarizeTextUseCase,
) -> None:
    # Given
    mock_summarize_text_usecase.execute = AsyncMock(return_value="summarize_result")

    # When
    response = client.post(
        "/summarize",
        json={
            "text": (
                "Photosynthesis is the process by which green plants and some other organisms "
                "use sunlight to synthesize foods with the help of chlorophyll."
            ),
        },
    )

    # Then
    assert response.status_code == 200
    assert response.json() == {
        "content": "summarize_result",
    }
    mock_summarize_text_usecase.execute.assert_awaited_once()


def test_summarize_missing_source_language(client: TestClient) -> None:
    # When
    response = client.post(
        "/summarize",
    )

    # Then
    assert response.status_code == 422  # Unprocessable Entity
