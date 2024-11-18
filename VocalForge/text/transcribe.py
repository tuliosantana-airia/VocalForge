from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI
from openai.types.audio.transcription_verbose import TranscriptionVerbose
from pydub import AudioSegment
from tqdm import tqdm
from whisper.normalizers.english import EnglishTextNormalizer

from .text_utils import get_files


class Transcribe:
    def __init__(
        self,
        input_dir: str,
        segmented_audio_dir: str,
        dataset_dir: str,
    ):
        self.input_dir = Path(input_dir)
        self.segmented_audio_dir = Path(segmented_audio_dir)
        self.dataset_dir = Path(dataset_dir)
        self.normalizer = EnglishTextNormalizer()
        self.client = OpenAI()
        self.input_files = get_files(str(self.input_dir), True, ".wav")

        self.transcriptions: List[TranscriptionVerbose] = []
        self.segmented_audios: List[str] = []
        self.segmented_texts: List[str] = []
        self.normalized_texts: List[str] = []

    def transcribe(self):
        for file in tqdm(
            self.input_files, total=len(self.input_files), desc="Transcribing"
        ):
            with open(file, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            self.transcriptions.append(transcript)

    def process(self):
        for transcript, file in tqdm(
            zip(self.transcriptions, self.input_files),
            total=len(self.transcriptions),
            desc="Segmenting and Normalizing",
        ):
            raw = AudioSegment.from_file(file, format="wav")
            for segment in transcript.segments:
                start = int(segment.start * 1000)
                end = int(segment.end * 1000)

                segmented_audio = raw[start:end]
                audio_path = str(
                    self.segmented_audio_dir / f"{Path(file).stem}_{segment.id}.wav"
                )
                segmented_audio.export(audio_path, format="wav")
                self.segmented_audios.append(audio_path)

                segmented_text = segment.text
                self.segmented_texts.append(segmented_text)

                normalized = self.normalizer(segmented_text)
                self.normalized_texts.append(normalized)

    def to_csv(self):
        pd.DataFrame(
            {
                "audio": self.segmented_audios,
                "text": self.segmented_texts,
                "normalized_text": self.normalized_texts,
            }
        ).to_csv(self.dataset_dir / "dataset.csv", index=False)
