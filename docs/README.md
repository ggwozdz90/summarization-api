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

### Windows Executable

Download the CPU version executable from [GitHub Releases](https://github.com/ggwozdz90/summarization-api/releases).

## Quick Start

### Prerequisites

Choose your preferred distribution:

- **Windows Executable**:
  - Windows 10 or later

- **Docker Images**:
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

### Using Windows Executable

1. Download from GitHub Releases
2. Run `summarization-api.exe`

## API Features

### Summarize Text

- Request:

    ```bash
    curl -X POST "http://localhost:8000/summarize" \
        -H "Content-Type: application/json" \
        -d '{"text": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."}'
    ```

- Response:

    ```json
    {
      "content": "Photosynthesis is how plants use sunlight to make food."
    }
    ```

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

Developer guide is available in [docs/DEVELOPER.md](DEVELOPER.md).

## Table of Contents

- [Summarization API](#summarization-api)
  - [Features](#features)
  - [Available Distributions](#available-distributions)
    - [Docker Images](#docker-images)
    - [Windows Executable](#windows-executable)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Using Docker](#using-docker)
    - [Using Docker Compose](#using-docker-compose)
    - [Using Windows Executable](#using-windows-executable)
  - [API Features](#api-features)
    - [Summarize Text](#summarize-text)
    - [Health Check](#health-check)
  - [Configuration](#configuration)
  - [Developer Guide](#developer-guide)
  - [Table of Contents](#table-of-contents)
