from pathlib import Path

from df.enhance import enhance, init_df, load_audio, save_audio
from pydub import AudioSegment
from pydub.effects import normalize
from tqdm import tqdm

from .audio_utils import get_files, create_samples


class ExportAudio:
    """
    A class for exporting audio files with various processing options.

    Args:
        input_dir (str, optional): The directory containing the input audio files. Defaults to None.
        output_dir (str, optional): The directory to export the formatted audio files. Defaults to None.
        noise_removed_dir (str, optional): The directory to export the noise-removed audio files. Defaults to None.
        normalization_dir (str, optional): The directory to export the normalized audio files. Defaults to None.
        sample_rate (int, optional): The sample rate of the audio files. Defaults to 22050.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        noise_removed_dir: str,
        normalization_dir: str,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.noise_removed_dir = Path(noise_removed_dir)
        self.normalization_dir = Path(normalization_dir)

        self.model, self.df_state, _ = init_df(config_allow_defaults=True)

    def noise_remove(self) -> None:
        files = get_files(self.output_dir)

        for file in tqdm(files, total=len(files), desc="Removing Noise"):
            audio, _ = load_audio(self.output_dir / file, sr=self.df_state.sr())
            enhanced = enhance(self.model, self.df_state, audio)
            save_audio(str(self.noise_removed_dir / file), enhanced, self.df_state.sr())

    def normalize(self):
        files = get_files(self.noise_removed_dir)

        for file in tqdm(files, total=len(files), desc="Normalizing"):
            audio = AudioSegment.from_file(self.noise_removed_dir / file, format="wav")
            normalized = normalize(audio)
            normalized.export(str(self.normalization_dir / file), format="wav")

    def create_samples(self, max_seconds: int = 120):
        create_samples(str(self.input_dir), str(self.output_dir), max_seconds, -1)
