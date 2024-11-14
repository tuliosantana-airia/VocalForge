from pathlib import Path

from pyannote.audio import Pipeline
from tqdm import tqdm

from .audio_utils import export_from_timestamps, get_files, get_timestamps


class VoiceDetection:
    def __init__(self, input_dir=None, output_dir=None, sample_dir=None, hparams=None):
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
        self.timelines = []
        self.timestamps = []
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

        if self.hparams is not None and self.is_hparams is False:
            self.pipeline.instantiate(self.hparams)
            self.is_hparams = True

        for file in tqdm(
            self.input_files, total=len(self.input_files), desc="Analyzing files"
        ):
            output = self.pipeline(file)
            self.timelines.append(output)

    def analyze_file(self, path):
        if self.hparams is not None and self.is_hparams is False:
            self.pipeline.instantiate(self.hparams)
            self.is_hparams = True
        """function to analyze a single file"""
        return self.pipeline(path)

    def find_timestamps(self):
        """
        This function processes speech metrics and returns timestamps
        of speech segments in the audio.

        Parameters:
        speech_metrics (list): list of speech metrics for audio file(s)

        Returns:
        Timestamps (list): list of speech timestamps for each audio file
        """
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

    def export(self):
        """
        Given a list of timestamps for each file, the function exports
        the speech segments from each raw file to a new file format wav.
        The new files are saved to a specified directory.
        """
        for index, file in tqdm(
            enumerate(self.input_files),
            total=len(self.input_files),
            desc="Exporting Speech Segments",
        ):
            base_file_name = Path(file).name
            export_from_timestamps(
                file,
                str(self.output_dir / base_file_name),
                self.timestamps[index],
            )

    def run(self):
        """runs the voice detection pipeline"""
        if list(self.input_dir.glob("*")) != []:
            self.analyze_folder()
        if self.timelines != []:
            self.find_timestamps()
        if self.timestamps != []:
            self.export()
        print("Analyzed files for voice detection")
