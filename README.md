# Oltralotus
Oltralotus is a lightweight, extensible server for getting up and running with Ultralytics models, inspired by what Ollama is for Llama.

### Building/Running
- locally:
    ```sh
    git clone https://github.com/alyshmahell/oltralotus
    cd oltralotus
    ./build
    ```
- from dockerhub:
    ```sh
    docker run -d --name oltralotus --gpus all -p 11535:11535 alyshmahell/oltralotus:latest
    ```
### Usage
- examples:
    ```sh
    python examples/client.py
    ```