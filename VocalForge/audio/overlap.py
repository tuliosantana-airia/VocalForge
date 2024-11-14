from pathlib import Path

from pyannote.audio import Pipeline
from tqdm import tqdm

from .audio_utils import export_from_timestamps, get_files, get_timestamps


class Overlap:
    def __init__(self, input_dir=None, output_dir=None, hparams=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_files = get_files(self.input_dir, True, ".wav")
        self.timelines = []
        self.timestamps = []
        self.hparams = hparams

        # Create a pipeline object using the pre-trained "pyannote/overlapped-speech-detection"
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/overlapped-speech-detection", use_auth_token=True
        )

        self.is_hparams = False
        if self.hparams is not None:
            self.pipeline.instantiate(self.hparams)
            self.is_hparams = True

    def analyze(self) -> list:
        """
        Analyzes overlapping speech in a set of speech audio files.

        Parameters:
        input_dir: (str) dir of input wav files
        """

        if self.hparams is not None and self.is_hparams is False:
            self.pipeline.instantiate(self.hparams)
            self.is_hparams = True

        for file in tqdm(
            self.input_files, total=len(self.input_files), desc="Analyzing files"
        ):
            overlap_timeline = self.pipeline(file)
            self.timelines.append(overlap_timeline)

    def find_timestamps(self):
        """
        This function processes speech metrics and returns timestamps
        of overlapping segments in the audio."""
        self.timestamps = []
        for fileindex in tqdm(
            range(len(self.input_files)),
            desc="Finding timestamps",
            total=len(self.input_files),
        ):
            timestamps = get_timestamps(self.timelines[fileindex])
            self.timestamps.append(timestamps)

    def update_timeline(self, new_timeline, index: int):
        """
        This function updates the timeline for a given file with the new timestamps due to finetuning
        """
        self.timelines[index] = new_timeline

        self.timestamps[index] = get_timestamps(new_timeline)

    def test_export(self):
        for index, file_str in tqdm(
            enumerate(self.input_files),
            total=len(self.input_files),
            desc="Exporting Speech Segments",
        ):
            file = Path(file_str)  # convert string to Path object
            base_file_name = file.name
            export_from_timestamps(
                file,
                self.output_dir / base_file_name,
                self.timestamps[index],
                combine_mode="time_between",
            )

    def run(self):
        """runs the overlap detection pipeline"""
        if any(Path(self.input_dir).iterdir()):
            self.analyze()
        if self.timelines:
            self.find_timestamps()
            print("Found timestamps")
        if self.timestamps:
            self.test_export()
        print("Analyzed files for voice detection")
