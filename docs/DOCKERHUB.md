# Summarization API

This project provides an API for summarizing text using state-of-the-art AI models. Text summarization is performed using Facebook's BART large CNN model.

## Features

- **RESTful API**: Exposes RESTful API endpoints
- **Summarization**: Text summarization using Facebook's BART large CNN model.
- **Configuration**: The repository includes a `.env` file that defines configurable environment variables.
- **Memory Optimization**: Models are loaded in separate processes and terminated after a configurable idle timeout to conserve RAM

## Available Distributions

### Docker Images

Available on [Docker Hub](https://hub.docker.com/r/ggwozdz/summarization-api):

- `version-cpu`: CPU-only version (fully tested and stable)
- `version-cuda124`: NVIDIA CUDA 12.4 support for GPU acceleration (proof-of-concept implementation)*
- `version-rocm62`: AMD ROCm 6.2 support for GPU acceleration (proof-of-concept implementation, requires build from source code)*
- `latest`: Points to latest CPU version

*Note on GPU Support: The current implementations of CUDA and ROCm support are provided as proof-of-concept solutions. While these implementations handle basic scenarios effectively, they haven't undergone comprehensive testing across all use cases. Users planning to utilize GPU acceleration may need to modify the Docker images to include additional environment-specific GPU support software. I recommend using the CPU version, which has been thoroughly tested and validated. The GPU implementations serve as a foundation for future development of more sophisticated functionality.

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/get-started/)

### Using Docker

- Run the following command to start the API server:

    ```bash
    docker run -d -p 8000:8000 \
      -e LOG_LEVEL=INFO \
      -e DEVICE=cpu \
      -e FASTAPI_HOST=0.0.0.0 \
      -e FASTAPI_PORT=8000 \
      -e MODEL_IDLE_TIMEOUT=60 \
      -e SUMMARIZATION_MODEL_NAME=facebook/bart-large-cnn \
      -e SUMMARIZATION_MODEL_DOWNLOAD_PATH=downloaded_summarization_models \
      -v ./volume/downloaded_summarization_models:/app/downloaded_summarization_models \
      ggwozdz/summarization-api:latest
    ```

### Using Docker Compose

- Create a `docker-compose.yml` file with the following content and run `docker-compose up`:

    ```yaml
    services:
      api:
        image: ggwozdz/summarization-api:latest
        environment:
          - LOG_LEVEL=INFO
          - DEVICE=cpu
          - FASTAPI_HOST=0.0.0.0
          - FASTAPI_PORT=8000
          - MODEL_IDLE_TIMEOUT=60
          - SUMMARIZATION_MODEL_NAME=facebook/bart-large-cnn
          - SUMMARIZATION_MODEL_DOWNLOAD_PATH=downloaded_summarization_models
        ports:
          - "8000:8000"
        volumes:
          - ./volume/downloaded_summarization_models:/app/downloaded_summarization_models
    ```

## API Features

### Summarize Text

- Request:

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/summarize' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "text_to_summarize": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington     Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
      "generation_parameters": { "num_beams": 3, "max_length": 100, "early_stopping": true }
    }'
    ```

- Response:

    ```json
    {
      "summary": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. It is the second tallest free-standing structure in France after the Millau Viaduct."
    }
    ```

#### Generation Parameters

The `generation_parameters` field in the request body allows you to specify the parameters which are described in the [Hugging Face Transformers documentation](https://huggingface.co/transformers/v2.11.0/model_doc/bart.html#transformers.BartForConditionalGeneration.generate):

### Health Check

- Request:

    ```bash
    curl -X GET "http://localhost:8000/healthcheck"
    ```

- Response:

    ```json
    {
      "status": "OK"
    }
    ```

## Configuration

The application uses a `.env` file or Docker Compose to define configurable environment variables. Below are the available configuration options:

- `LOG_LEVEL`: The logging level for the application. Supported levels are `NOTSET`, `DEBUG`, `INFO`, `WARN`, `WARNING`, `ERROR`, `FATAL`, and `CRITICAL`. The same log level will be applied to `uvicorn` and `uvicorn.access` loggers. Default is `INFO`.
- `DEVICE`: Device to run the models on (`cpu` or `cuda`). Default is `cpu`.
- `FASTAPI_HOST`: Host for the FastAPI server. Default is `127.0.0.1`.
- `FASTAPI_PORT`: Port for the FastAPI server. Default is `8000`.
- `SUMMARIZATION_MODEL_NAME`: Name of the summarization model to use. Supported models are `facebook/bart-large-cnn`. Default is `facebook/bart-large-cnn`.
- `SUMMARIZATION_MODEL_DOWNLOAD_PATH`: Path where summarization models are downloaded. Default is `downloaded_summarization_models`.
- `MODEL_IDLE_TIMEOUT`: Time in seconds after which the model will be unloaded if not used. Default is `60`.

## Developer Guide

Developer guide is available in [docs/DEVELOPER.md](https://github.com/ggwozdz90/summarization-api/blob/main/docs/DEVELOPER.md).
