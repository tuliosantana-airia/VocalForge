# Installation

1. Install FFMpeg

    ```bash
    sudo apt-get install ffmpeg
    ```

2. Create Conda Environment

    ```bash
    conda create -n vocalforge python=3.8
    ```

3. Install pynini, to use text normalization

    ```bash
    conda install -c conda-forge pynini
    ```

4. Install Cython

    ```bash
    pip install cython packaging
    ```

5. Install NeMo

    ```bash
    git clone --branch stable https://github.com/NVIDIA/NeMo
    cd NeMo
    pip install 'nemo_toolkit[asr]'
    pip install 'nemo_toolkit[nlp]'
    ```

6. Install Other Requirements

    ```bash
    cd ..
    pip install -r requirements.txt
    ````

7. Enter HuggingFace Token

    ```bash
    huggingface-cli login
    ```
