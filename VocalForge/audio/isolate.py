import shutil
from pathlib import Path
from uuid import uuid4

import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Annotation
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .audio_utils import get_files


class Isolate:
    def __init__(
        self,
        input_dir: str,
        verification_dir: str,
        output_dir: str,
        threshold: float = 0.1,
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

        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=True)
        self.inference = Inference(self.model, window="whole", device=self.device)
        self.embeddings = {}
        self.embeddings_files = {}

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

                combined.export(str(self.output_dir / f"{speaker}.wav"), format="wav")

    def _extract_folder_embeddings(self, folder_path: Path):
        for file in get_files(str(folder_path), True, ".wav"):
            embedding = self.inference(file)

            for key, value in self.embeddings.items():
                distance = cdist(value, embedding, metric="cosine")[0][0]
                if distance < self.threshold:
                    self.embeddings_files[key].append(file)
                    break
            else:  # new embedding
                emb_id = uuid4().hex
                self.embeddings[emb_id] = embedding
                self.embeddings_files[emb_id] = [file]

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
                splitted = value.rsplit("/", max_split=2)
                shutil.copy(
                    str(file), str(export_dir / Path(f"{splitted[-2]}_{splitted[-1]}"))
                )
