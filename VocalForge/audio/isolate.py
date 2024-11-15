import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import nemo.collections.asr as nemo_asr
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from tqdm import tqdm

from .audio_utils import get_files


class Isolate:
    def __init__(
        self,
        input_dir: str,
        verification_dir: str,
        output_dir: str,
        threshold: float = 0.7,
    ):
        self.input_dir = Path(input_dir)
        self.verification_dir = Path(verification_dir)
        self.output_dir = Path(output_dir)
        self.input_files = get_files(str(self.input_dir), True, ".wav")
        self.threshold = threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1", use_auth_token=True
        )
        self.pipeline.to(self.device)

        self.speaker_model: nemo_asr.models.EncDecSpeakerLabelModel = (
            nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                "nvidia/speakerverification_en_titanet_large", map_location=self.device
            )
        )
        self.target_embeddings: Dict[str, np.ndarray] = {}
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
        embedding = self.speaker_model.get_embedding(file_path).squeeze()
        self.target_embeddings[name] = embedding / torch.linalg.norm(embedding)
        self.embeddings_files[name] = []

    def _extract_folder_embeddings(self, folder_path: Path):
        sim_scores = {}
        for file in get_files(str(folder_path), True, ".wav"):
            embedding = self.speaker_model.get_embedding(file).squeeze()
            norm_emb = embedding / torch.linalg.norm(embedding)

            for key, value in self.target_embeddings.items():
                similarity_score = torch.dot(norm_emb, value) / (
                    (torch.dot(norm_emb, norm_emb) * torch.dot(value, value)) ** 0.5
                )
                similarity_score = (similarity_score + 1) / 2
                sim_scores[key] = similarity_score

        max_key = max(sim_scores, key=sim_scores.get)
        if sim_scores[max_key] >= self.threshold:
            self.embeddings_files[max_key].append(file)

    def group_audios_by_speaker(self):
        verification_folders = get_files(str(self.verification_dir))
        for folder in tqdm(
            verification_folders,
            total=len(verification_folders),
            desc="Grouping audios by speaker",
        ):
            folder_path = self.verification_dir / folder
            self._extract_folder_embeddings(folder_path)

        for key, value in self.embeddings_files.items():
            export_dir = self.output_dir / Path(key)
            if not export_dir.exists():
                export_dir.mkdir()

            for file in value:
                splitted = file.rsplit("/", maxsplit=2)
                shutil.copy(
                    str(file), str(export_dir / Path(f"{splitted[-2]}_{splitted[-1]}"))
                )
