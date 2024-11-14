# Installation

Supported only on Linux (no macOS or Windows)

1. Install FFMpeg

    ```bash
    sudo apt-get install ffmpeg
    ```

2. Create Conda Environment

    ```bash
    conda create -n vocalforge python=3.8
    ```

3. Install Cython

    ```bash
    pip install cython packaging
    ```

4. Install NeMo

    ```bash
    pip install nemo_toolkit[all]==1.21
    ```

5. Install Other Requirements

    ```bash
    pip install -r requirements.txt
    ````

6. Enter HuggingFace Token

    ```bash
    huggingface-cli login
    ```
