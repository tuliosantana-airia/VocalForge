from pathlib import Path
from typing import Optional

from datasets import Dataset
from pyannote.audio import Pipeline
from tqdm import tqdm

from .audio_utils import export_from_timestamps, get_files, get_timestamps


class VoiceDetection:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        batch_size: int = 1,
        sample_dir: Optional[str] = None,
        hparams: Optional[dict] = None,
    ):
        """
        Initializes a new instance of the VoiceDetection class.

        Parameters:
            input_dir (str): The directory containing the input audio files to analyze.
            output_dir (str): The directory where the output audio files will be saved.
            sample_dir (str): The directory containing sample audio files to analyze.
        """
        if sample_dir is not None:
            self.input_dir = Path(sample_dir)
        else:
            self.input_dir = Path(input_dir)
            self.output_dir = Path(output_dir)
        self.input_files = get_files(str(self.input_dir), True, ".wav")
        self.ds = Dataset.from_dict({"file": self.input_files})
        self.batch_size = batch_size
        self.hparams = hparams

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", use_auth_token=True
        )

        # instantiate the pipeline with hyperparameters if declared
        self.is_hparams = False
        if self.hparams is not None:
            self.pipeline.instantiate(self.hparams)
            self.is_hparams = True

    def analyze_folder(self):
        """
        Analyzes audio files in a folder and performs voice activity detection (VAD)
        on the audio files. It uses the 'pyannote.audio' library's pre-trained 'brouhaha' model for the analysis.
        """
        self.ds.map(self._analyze_file, num_proc=self.batch_size)

    def _analyze_file(self, example):
        timeline = self.pipeline(example["file"])
        example["timestamps"] = get_timestamps(timeline)

        return example

    def export(self):
        """
        Given a list of timestamps for each file, the function exports
        the speech segments from each raw file to a new file format wav.
        The new files are saved to a specified directory.
        """
        for example in tqdm(
            self.ds,
            total=len(self.ds),
            desc="Exporting Speech Segments",
        ):
            base_file_name = Path(example["file"]).name
            export_from_timestamps(
                example["file"],
                str(self.output_dir / base_file_name),
                example["timestamps"],
            )

    def run(self):
        """runs the voice detection pipeline"""
        if list(self.input_dir.glob("*")) != []:
            self.analyze_folder()
            self.export()

        print("Analyzed files for voice detection")
