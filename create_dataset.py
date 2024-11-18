import datetime as dt
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

from VocalForge.audio import ExportAudio, Isolate, Overlap, VoiceDetection
from VocalForge.audio.audio_utils import create_core_folders, download_videos
from VocalForge.text import Transcribe


class Folders(Enum):
    RAW_AUDIO = "RawAudio"
    VD = "VD"
    OVERLAP = "Overlap"
    VERIFICATION = "Verification"
    ISOLATED = "Isolated"
    NOISE_REMOVED = "NoiseRemoved"
    NORMALIZED = "Normalized"
    WHISPER_SAMPLES = "WhisperSamples"
    FINAL_SEGMENTS = "FinalSegments"
    DATASET = "Dataset"

    @classmethod
    def as_list(cls):
        return [k.value for k in cls]


def main():
    load_dotenv()

    parser = ArgumentParser()
    parser.add_argument(
        "--url", type=str, required=True, help="URL of the youtube playlist or video"
    )
    parser.add_argument(
        "--embedding", type=str, required=True, help="Embedding of the speaker"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25, help="Threshold for embedding model"
    )
    args = parser.parse_args()

    root_path = Path.cwd()
    work_path = root_path / "work" / dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    work_path.mkdir(parents=True, exist_ok=True)
    folder_names = Folders.as_list()

    create_core_folders(folder_names, str(work_path))
    download_videos(url=args.url, out_dir=str(work_path / Folders.RAW_AUDIO.value))

    vd = VoiceDetection(
        str(work_path / Folders.RAW_AUDIO.value), str(work_path / Folders.VD.value)
    )
    vd.run()

    overlap = Overlap(
        str(work_path / Folders.VD.value), str(work_path / Folders.OVERLAP.value)
    )
    overlap.run()

    isolation = Isolate(
        str(work_path / Folders.OVERLAP.value),
        str(work_path / Folders.VERIFICATION.value),
        str(work_path / Folders.ISOLATED.value),
    )
    embedding_name = Path(args.embedding).stem
    isolation.isolate_speakers()
    isolation.create_target_embedding(args.embedding, embedding_name)
    isolation.group_audios_by_speaker(args.threshold)

    exporter = ExportAudio(
        str(work_path / Folders.ISOLATED.value / embedding_name),
        str(work_path / Folders.WHISPER_SAMPLES.value),
        str(work_path / Folders.NOISE_REMOVED.value),
        str(work_path / Folders.NORMALIZED.value),
    )
    exporter.noise_remove()
    exporter.normalize()
    exporter.create_samples(max_seconds=120)

    transcriber = Transcribe(
        str(work_path / Folders.WHISPER_SAMPLES.value),
        str(work_path / Folders.FINAL_SEGMENTS.value),
        str(work_path / Folders.DATASET.value),
    )
    transcriber.transcribe()
    transcriber.process()
    transcriber.to_csv()


if __name__ == "__main__":
    main()
