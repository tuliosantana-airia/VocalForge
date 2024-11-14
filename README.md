# An End-to-End Toolkit for Voice Datasets

`VocalForge` is an open-source toolkit written in Python 🐍 that is meant to cut down the time to create datasets for, TTS models, hotword detection models, and more so you can spend more time training, and less time sifting through audio data.

Using [Nvidia's NEMO](https://github.com/NVIDIA/NeMo), [PyAnnote](https://github.com/pyannote/pyannote-audio), [CTC segmentation](https://github.com/lumaku/ctc-segmentation), [OpenAI's Whisper](https://github.com/openai/whisper), this repo will take you from raw audio to a fully formatted dataset, refining both the audio and text automatically.

> *NOTE*: While this does reduce time on spent on dataset curation, verifying the output at each step is important as it isn't perfect

![a flow chart of how this repo works](https://github.com/rioharper/VocalForge/blob/main/media/join_processes.svg?raw=true)

## Features

### `audio_demo.ipynb`

- ⬇️ **Download audio** from a YouTube playlist (perfect for podcasts/interviews) OR input your own raw audio files (wav format)

- 🎵 **Remove Non Speech Data**

- 🗣🗣 **Remove Overlapping Speech**

- 👥 **Split Audio File Into Speakers**

- 👤 **Isolate the same speaker across multiple files (voice verification)**

- 🧽 **Use DeepFilterNet to reduce background noise**

- 🧮 **Normalize Audio**

- ➡️ **Export with user defined parameters**

### `text_demo.ipynb`

- 📜 **Batch transcribe text using OpenAI's Whisper**

- 🧮 **Run [text normalization](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/text_normalization/wfst/wfst_text_normalization.html)**

- 🫶 **Use CTC segmentation to line up text to audio**

- 🖖 **Split audio based on quality of CTC segmentation confidence**

- ✅ **Generate a metadata.csv and dataset in the format of LJSpeech**

### `VCAuditor`

- 📊 **View the waveforms of your audio using a custom version of [WaveSurfer.js](https://wavesurfer-js.org/)**

- 📋**Verify the text created with Whisper Transcription**

- ✍️ **Edit and verify VocalForge generated timestamps to align with the audio**

- 🔃 **Sort by confidence to delete or modify the least confident alignments**
  
![an exmaple of VCAuditor](https://github.com/rioharper/VocalForge/blob/main/media/auditor_example.png?raw=true)

## Setup/Requirements

Please follow the instructions on [INSTALL.md](INSTALL.md)

- Python 3.8 has been tested, newer versions should work

- CUDA is required to run all models

- a [Hugging Face account](https://huggingface.co/) is required (it's free and super helpful!)

Pyannote models need to be "signed up for" in Hugging Face for research purposes. Don't worry, all it asks for is your purpose, website and organization.

![an example of signing up for a model](https://github.com/rioharper/VocalForge/blob/main/media/huggingface.png?raw=true)

The following models will have to be manually visited and given the appropriate info:

- [VAD model](https://huggingface.co/pyannote/voice-activity-detection)

- [Overlapped Speech Detection](https://huggingface.co/pyannote/overlapped-speech-detection)

- [Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization)

- [Embedding](https://huggingface.co/pyannote/embedding)

- [Segmentation](https://huggingface.co/pyannote/segmentation)

### **Setting up VCAuditor**

VCAuditor Uses NodeJS, and is a browser based GUI tool. To allow for the launching of js files, we will need to install NodeJS into our conda environment and install the necessary libraries:

`conda install -c conda-forge nodejs`
`cd VCAuditor`
`npm install`
`npm run`

And with this, localhost:3000 should have VCAuditor online! To load up your VocalForge project, make sure to include the text files under the "Segments" folder, and the audio files from the "Input_Audio" folder.  

Some helpful tips for you as you comb through your data:

- To play audio, press the spacebar. If for some reason the spacebar does not start the audio, switch tabs, and then go back to the tab you were on. This is a browser security issue, but that should fix it.
- To loop a region, shift-click the region of your choosing, and it will play from the exact start time.
- When you feel like a region is verified, press the checkmark on the region and it will mark it as manually verified
- If a region is so off that you can't simply fix it, you can press the trash icon on the region or the table to remove it
- You can drag around the table window if it is blocking something

Once you are ready to export a verified segment file, then press the export button on the table, and a file will download. This file does not include the multiple normalization text versions that the original file had.

## API Example

```python
from VocalForge.text.normalize_text import NormalizeText

normalize = NormalizeText(
    input_dir= os.path.join(work_path, 'transcription'),
    out_dir= os.path.join(work_path, 'processed'),
    audio_dir= os.path.join(work_path, 'input_audio'),
)

normalize.run()
```
