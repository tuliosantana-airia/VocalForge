from pydub import AudioSegment
from pathlib import Path
import yt_dlp
import natsort
from typing import List, Tuple


def download_videos(url: str, out_dir: str):
    """This function downloads audio from a youtube playlist and saves it to disk in the .wav format.
    If the audio is longer than 1 hour, it is split into smaller clips and saved to disk.

     Inputs:
     - url: a string representing the url of the youtube playlist or youtube video.
     - out_dir: a string representing the directory path to save the downloaded audio.

     Outputs:
     - None, but audio clips are saved to disk in the .wav format.
    """
    out_path = Path(out_dir)
    ydl_opts = {
        "format": "wav/bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "outtmpl": str(out_path / "%(title)s.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(url)

    # 5 second blank as pyannote doesnt seem to process first ~3 seconds of audio
    blank = AudioSegment.silent(duration=5000)

    # split if audio is above 1 hour
    for filename in out_path.iterdir():
        file = AudioSegment.from_file(filename)
        if len(file) > 3500000:
            slices = file[::3500000]
            for index, slice in enumerate(slices):
                slice.export(out_path / f"{filename.stem}_{index}.wav", format="wav")
            filename.unlink()
        # add 5 second blank at start

    for filename in out_path.iterdir():
        file = AudioSegment.from_file(filename)
        if filename.stat().st_size > 500000000:
            slices = file[:: int((file.duration_seconds * 1000) / 2)]
            for index, slice in enumerate(slices):
                slice = blank + slice
                slice.export(out_path / f"{filename.stem}_{index}.wav", format="wav")
            filename.unlink()
        else:
            file = blank + file
            file.export(filename, format="wav")

    for count, filename in enumerate(out_path.iterdir()):
        dst = f"DATA{count}.wav"
        dst = out_path / dst
        filename.rename(dst)


def split_files(folder: str, dir: str, duration: int):
    """This function splits audio files in the .wav format located in the
    specified folder and saves the clips in the same folder.

     Inputs:
     - folder: a string representing the name of the folder containing the audio files.
     - dir: a string representing the directory path containing the folder.
     - duration: ms of the duration of sample clips.

     Outputs:
     - None, but audio clips are saved to disk in the .wav format.

    """

    folder_dir = Path(dir, folder)
    for file in get_files(str(folder_dir), ext=".wav"):
        file_path = folder_dir / file
        raw = AudioSegment.from_file(file_path, format="wav")
        for index, clip in enumerate(raw[::duration]):
            clip_dir = folder_dir / file_path.stem / (f"{file_path.stem}_{index}.wav")
            clip.export(clip_dir, format="wav")


def get_files(dir: str, full_dir: bool = False, ext: str = None) -> list:
    """
    Retrieves a list of files in a directory, sorted in natural order.

    Parameters:
        dir (str): A string representing the directory path to search for files.
        full_dir (bool): A boolean indicating whether to return the full directory path or just the file name. Default is False.
        ext (str): A string representing the file extension to filter by. If None, all files are returned. Default is None.

    Returns:
        files (list): A list of file names sorted in natural order.

    Example:
        get_files('/home/user/documents', True, '.txt')
        Returns:
        ['/home/user/documents/file1.txt', '/home/user/documents/file2.txt']
    """
    dir_path = Path(dir)
    files = []
    for file in dir_path.iterdir():
        if ext is not None:
            if file.suffix == ext:
                if full_dir:
                    files.append(str(file))
                else:
                    files.append(file.name)
        else:
            files.append(file.name)
    files = natsort.natsorted(files)
    return files


def create_core_folders(folders: list, workdir: str):
    """
    This function creates a list of folders in a specified directory if they do not already exist.

    Parameters:
        folders (list): A list of folder names to be created.
        workdir (str): A string representing the directory path where the folders will be created.

    Returns:
        None

    Example:
        create_core_folders(['raw', 'processed'], '/home/user/documents')
        Creates the folders 'raw' and 'processed' in the directory '/home/user/documents'.
    """
    workdir_path = Path(workdir)
    for folder in folders:
        folder_path = workdir_path / folder
        folder_path.mkdir(exist_ok=True)


def create_samples(length: int, input_dir: str, output_dir: str) -> None:
    """This function creates audio samples of a specified length from audio files
       in the .wav format located in a specified raw directory.

    Parameters:
        length (int): An integer representing the length in seconds of the samples to be created.
        input_dir (str): A string representing the folder where raw wav files are located.
        output_dir (str): A string representing the location for output sample wav files.

    Returns:
        None, but audio samples are saved to disk in the .wav format.

    Example:
        create_samples(5, '/home/user/documents/raw', '/home/user/documents/samples')
        Creates audio samples of 5 seconds from audio files in the .wav format located in the
        '/home/user/documents/raw' directory and saves them to the '/home/user/documents/samples'
        directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    rawfiles = get_files(input_path, ".wav")

    for file in rawfiles:
        raw_data = AudioSegment.from_file(input_path / file, format="wav")
        entry = raw_data[: length * 1000]
        nfilename = output_path / file
        entry.export(nfilename, format="wav")


# function with timeline being a pyannote.core.annotation.Annotation object
def get_timestamps(timeline) -> List[Tuple[int, int]]:
    """This function takes in a pyannote.core.annotation.Annotation object and returns a list of timestamps
    where each timestamp is a tuple containing the start and end time of a period.

     Inputs:
     - timeline: a pyannote.core.annotation.Annotation object.

     Outputs:
     - A list of timestamps where each timestamp is a tuple containing the start and end time of a period.
    """

    timestamps = []
    for segment in timeline.get_timeline():
        timestamps.append((segment.start, segment.end))
    return timestamps


def remove_short_timestamps(
    timestamps: List[Tuple[int, int]], min_duration: int
) -> List[Tuple[int, int]]:
    """
    Removes timestamps that are too short from a list of timestamps.

    Parameters:
    timestamps (List[Tuple[int, int]]): List of timestamps. Each timestamp is a tuple containing
                 the start and end time of a period.
    min_duration (int): The minimum duration in seconds for a timestamp to be included in the output.

    Returns:
    List[Tuple[int, int]]: List of timestamps with short timestamps removed.
    """
    return [(start, end) for start, end in timestamps if end - start > min_duration]


def concentrate_timestamps(
    timestamps: List[Tuple[int, int]], min_duration: int
) -> List[Tuple[int, int]]:
    """This function takes in a list of timestamps and returns a condensed list of
    timestamps where timestamps are merged that are close to each other.

     Inputs:
     - timestamps: a list of timestamp tuples or a list of single timestamps.
     - min_duration: an integer representing the minimum duration between timestamps
       to be combined.

     Outputs:
     - A list of condensed timestamps where timestamps that are within
       {min_duration} of each other have been merged into a single entry.
    """

    concatenated_timestamps = []
    current_start = timestamps[0][0]
    current_stop = timestamps[0][1]
    for i in range(1, len(timestamps)):
        if timestamps[i][0] - current_stop <= min_duration:
            current_stop = timestamps[i][1]
        else:
            concatenated_timestamps.append([current_start, current_stop])
            current_start = timestamps[i][0]
            current_stop = timestamps[i][1]
    concatenated_timestamps.append([current_start, current_stop])
    return concatenated_timestamps


def calculate_duration(timestamps: List[Tuple[int, int]]) -> float:
    """pretty self expainatory"""
    duration = 0
    for timestamp in timestamps:
        duration += timestamp[1] - timestamp[0]
    return duration


def find_original_duration(input_file_dir: str) -> float:
    """pretty self expainatory"""
    raw = AudioSegment.from_file(input_file_dir, format="wav")
    return raw.duration_seconds


def find_duration_diff(
    new_timestamps: List[Tuple[int, int]], original_duration: float
) -> float:
    """pretty self expainatory"""
    new_duration = calculate_duration(new_timestamps)
    # if abs(original_duration - new_duration) > original_duration * 0.8:
    #     return abs(new_duration - original_duration)
    return abs(original_duration - new_duration)


def export_from_timestamps(
    input_file_dir,
    export_file_dir,
    timestamps: List[Tuple[int, int]],
    combine_mode: str = "timestamps",
) -> None:
    """'Exports audio from timestamps to a new file.

    Inputs:
    - input_file_dir: a string representing the directory path of the input audio file.
    - export_file_dir: a string representing the directory path to save the exported audio file.
    - timestamps: a list of timestamp tuples or a list of single timestamps. **NOTE: timestamps must be in seconds**
    - combine_mode: a string representing the mode of combination. Valid values are 'timestamps' (default) and 'time_between'.
    """

    new_file = AudioSegment.empty()
    raw = AudioSegment.from_file(input_file_dir, format="wav")

    if combine_mode == "timestamps":
        if len(timestamps) == 0:
            return
        for timestamp in timestamps:
            new_file += raw[timestamp[0] * 1000 : timestamp[1] * 1000]

    elif combine_mode == "time_between":
        if len(timestamps) == 0:
            raw.export(export_file_dir, format="wav")

        start = 0
        for timestamp in timestamps:
            new_file += raw[start : timestamp[0] * 1000]
            start = timestamp[1] * 1000
        new_file += raw[start:]

    if len(new_file) > 1000:
        new_file.export(export_file_dir, format="wav")
    else:
        print(f"{input_file_dir} doesn't have enough clean audio to export")
