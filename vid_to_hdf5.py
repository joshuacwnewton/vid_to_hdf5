"""
    vid_to_hdf5.py: Basic utility to extract frames from video and store
    them in an hdf5 container.
"""

# TODO: Make use of proper logging rather than print statements
# TODO: Write more detailed documentation (usage, usecases, etc.)
# TODO: Expose more configuration (frame count, compression opts)
# TODO: Refactor main into more modular structure
# TODO: Package and distribute on PyPi

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


# Relevant video properties for cv2's VidCap class
CV2_PROPIDS = [
    "CAP_PROP_FRAME_WIDTH",
    "CAP_PROP_FRAME_HEIGHT",
    "CAP_PROP_FPS",
    "CAP_PROP_FOURCC",
    "CAP_PROP_FRAME_COUNT",
    "CAP_PROP_FORMAT",
    "CAP_PROP_MODE",
    "CAP_PROP_CONVERT_RGB",
    "CAP_PROP_BUFFERSIZE"
]


def main(filepaths):
    for filepath in filepaths:
        # Ensure that video filepath is a Path object
        filepath = Path(filepath)

        # Initialize video stream
        stream = cv2.VideoCapture(str(filepath))
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

        # Variable-length uint8 datatype for encoded png streams
        dt = h5py.vlen_dtype(np.dtype('uint8'))

        # Initialize hdf5 dataset
        f = h5py.File(filepath.parent / f"{filepath.stem}.h5", "w")
        dset = f.create_dataset('VideoFrames', (total_frames,), dtype=dt)

        # Set video properties as dataset attributes
        for prop in CV2_PROPIDS:
            dset.attrs[prop] = stream.get(eval("cv2."+prop))

        frame_number = 0
        while frame_number < total_frames:
            # Determine attributes associated with frame
            frame_number = int(stream.get(cv2.CAP_PROP_POS_FRAMES))

            # Attempt to load new frame and compress it using png encoding
            success, frame = stream.read()
            success, framebuf = cv2.imencode(".png", frame)

            # Store compressed frame in variable length hdf5 dataset
            dset[frame_number] = np.transpose(framebuf)

            status_update(frame_number, total_frames)


def status_update(frames_processed, total_frames):
    """Provide frequent status updates on how many frames have been
    processed"""
    if frames_processed % 25 is 0 and frames_processed is not 0:
        sys.stdout.write("\r[-]     {0}/{1} frames processed."
                         .format(frames_processed, total_frames))
        sys.stdout.flush()


def parse_filepaths():
    """Parse all command line arguments as filepaths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", nargs="*")
    args = parser.parse_args()

    args.filepaths = [Path(filepath).resolve() for filepath in args.filepaths]

    return args.filepaths


def validate_filepaths(filepaths):
    """Ensure that file path points to a valid file."""

    if type(filepaths) is not list:
        filepaths = [filepaths]

    for filepath in filepaths:
        filepath = Path(filepath)

        if not Path.is_file(filepath):
            sys.stderr.write("[!] Error: {} does not point to a valid file."
                             .format(filepath.name))
            sys.exit()


def validate_video_files(video_filepaths):
    """Ensure that frames can be read from video file."""

    if type(video_filepaths) is not list:
        video_filepaths = [video_filepaths]

    for video_filepath in video_filepaths:
        vidcap = cv2.VideoCapture(str(video_filepath))
        retval, _ = vidcap.read()

        if retval is False:
            sys.stderr.write("[!] Error: Unable to read frames from {}."
                             .format(video_filepath.name))
            sys.exit()

        vidcap.release()


if __name__ == "__main__":
    paths = parse_filepaths()

    validate_filepaths(paths)
    validate_video_files(paths)

    main(paths)
