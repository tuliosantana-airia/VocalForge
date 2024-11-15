import shutil
from pathlib import Path
from typing import Dict, List

import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from speechbrain.inference.speaker import SpeakerRecognition
from tqdm import tqdm

from .audio_utils import get_files


class Isolate:
    def __init__(
        self,
        input_dir: str,
        verification_dir: str,
        output_dir: str,
    ):
        self.input_dir = Path(input_dir)
        self.verification_dir = Path(verification_dir)
        self.output_dir = Path(output_dir)
        self.input_files = get_files(str(self.input_dir), True, ".wav")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1", use_auth_token=True
        )
        self.pipeline.to(self.device)

        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda"},
        )
        self.target_embeddings: Dict[str, torch.Tensor] = {}
        self.embeddings_files: Dict[str, List[str]] = {}

    def isolate_speakers(self) -> list:
        for file in tqdm(
            self.input_files,
            total=len(self.input_files),
            desc="Isolating Speakers in each file",
        ):
            diarization: Annotation = self.pipeline(file)
            audio = AudioSegment.from_file(file, format="wav")
            speaker_segments = {}

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = int(turn.start * 1000)
                end_time = int(turn.end * 1000)
                segment = audio[start_time:end_time]

                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append(segment)

            for speaker, segments in speaker_segments.items():
                combined = sum(segments)
                folder_name = Path(file).stem
                speaker_dir = self.verification_dir / folder_name

                if not speaker_dir.exists():
                    speaker_dir.mkdir()

                combined.export(str(speaker_dir / Path(f"{speaker}.wav")), format="wav")

    def create_target_embedding(self, file_path: str, name: str):
        waveform = self.verification.load_audio(file_path).unsqueeze(0)
        self.target_embeddings[name] = self.verification.encode_batch(waveform)
        self.embeddings_files[name] = []

    def _extract_folder_embeddings(self, folder_path: Path, threshold: float = 0.25):
        scores = {k: -1 for k in self.target_embeddings}
        files = {k: None for k in self.target_embeddings}
        for file in get_files(str(folder_path), True, ".wav"):
            waveform = self.verification.load_audio(file).unsqueeze(0)
            embedding = self.verification.encode_batch(waveform)

            for key, value in self.target_embeddings.items():
                score = self.verification.similarity(embedding, value)
                if score > scores[key]:
                    scores[key] = score
                    files[key] = file

        for k in scores:
            if scores[k] > threshold:
                self.embeddings_files[k].append(files[k])

    def group_audios_by_speaker(self, threshold: float = 0.25):
        verification_folders = get_files(str(self.verification_dir))
        for folder in tqdm(
            verification_folders,
            total=len(verification_folders),
            desc="Grouping audios by speaker",
        ):
            folder_path = self.verification_dir / folder
            self._extract_folder_embeddings(folder_path, threshold=threshold)

        for key, value in self.embeddings_files.items():
            export_dir = self.output_dir / Path(key)
            if not export_dir.exists():
                export_dir.mkdir()

            for file in value:
                splitted = file.rsplit("/", maxsplit=2)
                shutil.copy(
                    str(file), str(export_dir / Path(f"{splitted[-2]}_{splitted[-1]}"))
                )
