"""
Device file reader and high-level processing pipeline.

This module provides functions to read accelerometer data from various device
formats and apply a full processing pipeline including filtering, calibration,
nonwear detection, and resampling.

Supported Devices
-----------------
- Axivity AX3/AX6 (.cwa files)
- Actigraph (.gt3x files)
- GENEActiv (.bin files)
- Matrix (.bin files)

Main Functions
--------------
read_device : Read device file and apply processing pipeline
process : Apply processing pipeline to existing DataFrame

The module uses Java-based parsers for most device types (via subprocess calls),
and a pure Python parser for Matrix devices. All parsers output data in a
consistent pandas DataFrame format with a DateTimeIndex.

Notes
-----
- Files can be compressed (.gz, .zip) and will be automatically decompressed
- Processing is memory-efficient using chunked operations for large files
- Temporary files are created during parsing and cleaned up automatically
"""

import gzip
import math
import os
import pathlib
import shutil
import struct
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime
from numbers import Real
from typing import IO, Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from actipy import matrix_reader
from actipy import processing as P

__all__ = ['read_device', 'process']

Info = Dict[str, Any]
Frequency = Optional[Union[int, float, bool]]
ResampleFrequency = Optional[Union[Literal['uniform'], int, float, bool]]
Timestamp = Optional[Union[str, datetime]]


def _validate_resample_frequency(resample_hz: object) -> None:
    if resample_hz is None or isinstance(resample_hz, bool):
        return
    if isinstance(resample_hz, str):
        if resample_hz == 'uniform':
            return
        raise ValueError(
            "resample_hz must be 'uniform', True, a positive finite number, "
            "None, or False."
        )
    if (
        not isinstance(resample_hz, Real)
        or not math.isfinite(resample_hz)
        or resample_hz <= 0
    ):
        raise ValueError(
            "resample_hz must be 'uniform', True, a positive finite number, "
            "None, or False."
        )


def read_device(input_file: str,
                lowpass_hz: Frequency = 20,
                calibrate_gravity: bool = True,
                detect_nonwear: bool = True,
                resample_hz: ResampleFrequency = 'uniform',
                start_time: Timestamp = None,
                end_time: Timestamp = None,
                skipdays: int = 0,
                cutdays: int = 0,
                start_first_complete_minute: bool = False,
                calibrate_gravity_kwargs: Optional[Dict[str, Any]] = None,
                flag_nonwear_kwargs: Optional[Dict[str, Any]] = None,
                verbose: bool = True) -> Tuple[pd.DataFrame, Info]:
    """
    Read and process accelerometer device file.

    This is the main entry point for reading device files. It performs the full
    processing pipeline: file parsing, quality control, lowpass filtering,
    gravity calibration, nonwear detection, and resampling.

    :param input_file: Path to accelerometer file (.cwa, .gt3x, .bin).
        Compressed files (.gz, .zip) are automatically decompressed.
    :type input_file: str
    :param lowpass_hz: Cutoff frequency (Hz) for Butterworth lowpass filter.
        Defaults to 20 Hz. Pass None or False to disable filtering.
    :type lowpass_hz: int or False, optional
    :param calibrate_gravity: Whether to perform gravity calibration using the
        method of van Hees et al. 2014. Defaults to True.
    :type calibrate_gravity: bool, optional
    :param detect_nonwear: Whether to detect and flag non-wear periods (long
        stationary periods). Defaults to True.
    :type detect_nonwear: bool, optional
    :param resample_hz: Target frequency (Hz) to resample the signal. If
        "uniform", uses the device's sample rate to fix sampling errors. Pass
        None or False to disable. Defaults to "uniform".
    :type resample_hz: "uniform" or int or float or bool, optional
    :param start_time: Start time to read data (ISO format: "YYYY-MM-DD HH:MM:SS").
        Pass None to read from the beginning. Defaults to None.
    :type start_time: str or datetime, optional
    :param end_time: End time to read data (ISO format: "YYYY-MM-DD HH:MM:SS").
        Pass None to read until the end. Defaults to None.
    :type end_time: str or datetime, optional
    :param skipdays: Number of days to skip from the beginning. Defaults to 0.
    :type skipdays: int, optional
    :param cutdays: Number of days to cut from the end. Defaults to 0.
    :type cutdays: int, optional
    :param start_first_complete_minute: Whether to start data from the first
        complete minute (with 1 second tolerance). Useful for aligning data to
        minute boundaries. Defaults to False.
    :type start_first_complete_minute: bool, optional
    :param calibrate_gravity_kwargs: Additional keyword arguments for the
        calibrate_gravity function (e.g., {'stdtol_min': 0.01}). Defaults to None.
    :type calibrate_gravity_kwargs: dict, optional
    :param flag_nonwear_kwargs: Additional keyword arguments for the flag_nonwear
        function (e.g., {'patience': '60m'}). Defaults to None.
    :type flag_nonwear_kwargs: dict, optional
    :param verbose: Print progress messages. Defaults to True.
    :type verbose: bool, optional
    :return: A tuple (data, info) where:

        - **data** (pandas.DataFrame): Processed acceleration time-series with
          DateTimeIndex and columns for x, y, z acceleration (in g), plus
          optional temperature and light data depending on device type.
        - **info** (dict): Processing metadata including device information,
          quality metrics, calibration results, and processing parameters.
          See GLOSSARY.md for complete field descriptions.
    :rtype: tuple(pandas.DataFrame, dict)

    Examples
    --------
    Basic usage with default processing:

    >>> import actipy
    >>> data, info = actipy.read_device("sample.cwa.gz")
    >>> print(data.head())
                                 x         y         z  temperature    light
    time
    2014-05-07 13:29:50.430 -0.514     0.070     1.672        20.0    78.42
    2014-05-07 13:29:50.440 -0.234    -0.587     0.082        20.0    78.42
    ...

    With custom processing options:

    >>> data, info = actipy.read_device(
    ...     "sample.cwa.gz",
    ...     lowpass_hz=20,
    ...     calibrate_gravity=True,
    ...     detect_nonwear=True,
    ...     resample_hz=50
    ... )

    Time filtering and alignment:

    >>> data, info = actipy.read_device(
    ...     "sample.cwa.gz",
    ...     start_time="2014-05-07 18:00:00",
    ...     end_time="2014-05-09 18:00:00",
    ...     skipdays=1,
    ...     cutdays=2,
    ...     start_first_complete_minute=True
    ... )

    Custom calibration parameters:

    >>> data, info = actipy.read_device(
    ...     "sample.cwa.gz",
    ...     calibrate_gravity=True,
    ...     calibrate_gravity_kwargs={'stdtol_min': 0.01, 'calib_cube': 0.5}
    ... )

    See Also
    --------
    process : Apply processing pipeline to existing DataFrame
    actipy.processing : Individual processing functions for fine-grained control

    Notes
    -----
    - Supported formats: Axivity (.cwa), Actigraph (.gt3x), GENEActiv (.bin), Matrix (.bin)
    - Processing is memory-efficient using chunked operations for large files
    - The info dict is progressively updated as each processing step completes
    - Non-wear periods are set to NaN if detect_nonwear=True
    """

    _validate_resample_frequency(resample_hz)
    timer = Timer(verbose)

    data, info = _read_device(input_file, verbose)

    # Filter data by start/end time, if specified
    if start_time is not None:
        data = data.loc[cast(Any, start_time):]
    if end_time is not None:
        data = data.loc[:cast(Any, end_time)]

    # Skip/cut days, if specified
    if skipdays > 0:
        data = data.loc[data.index[0] + pd.Timedelta(days=skipdays):]
    if cutdays > 0:
        data = data.loc[:data.index[-1] - pd.Timedelta(days=cutdays)]

    # NOTE: Using process() increases data ref count by 1, which increases
    # memory. So instead we just do everything here.

    # Basic quality control
    timer.start("Quality control...")
    data, info_qc = P.quality_control(data, info['SampleRate'])
    info_qc['ReadErrors'] += info['ReadErrors']
    info.update(info_qc)
    timer.stop()

    if len(data) == 0:
        print("File is empty. No data to process.")
        return data, info

    if lowpass_hz not in (None, False):
        timer.start("Lowpass filter...")
        data, info_lowpass = P.lowpass(data, info['SampleRate'], cast(float, lowpass_hz))
        info.update(info_lowpass)
        timer.stop()

    if calibrate_gravity:
        timer.start("Gravity calibration...")
        calib_kwargs = calibrate_gravity_kwargs or {}
        data, info_calib = P.calibrate_gravity(data, return_coeffs=False, **calib_kwargs)
        info.update(info_calib)
        timer.stop()

    if detect_nonwear:
        timer.start("Nonwear detection...")
        nonwear_kwargs = flag_nonwear_kwargs or {}
        data, info_nonwear = P.flag_nonwear(data, **nonwear_kwargs)
        info.update(info_nonwear)
        timer.stop()

    if resample_hz is not None and resample_hz is not False:
        timer.start("Resampling...")
        if resample_hz == 'uniform' or resample_hz is True:
            data, info_resample = P.resample(data, info['SampleRate'], start_first_complete_minute=start_first_complete_minute)
        else:
            data, info_resample = P.resample(data, cast(float, resample_hz), start_first_complete_minute=start_first_complete_minute)
        info.update(info_resample)
        timer.stop()

    return data, info


def process(data: pd.DataFrame, sample_rate: float,
            lowpass_hz: Frequency = 20,
            calibrate_gravity: bool = True,
            detect_nonwear: bool = True,
            resample_hz: ResampleFrequency = 'uniform',
            start_first_complete_minute: bool = False,
            calibrate_gravity_kwargs: Optional[Dict[str, Any]] = None,
            flag_nonwear_kwargs: Optional[Dict[str, Any]] = None,
            verbose: bool = True) -> Tuple[pd.DataFrame, Info]:
    """
    Apply processing pipeline to acceleration time-series DataFrame.

    This function applies the same processing steps as read_device() but to an
    existing pandas DataFrame. Useful for processing custom CSV files or data
    from sources not directly supported by read_device().

    :param data: A pandas.DataFrame of acceleration time-series. Must contain
        at least columns 'x', 'y', 'z' (acceleration in g) and the index must
        be a DateTimeIndex.
    :type data: pandas.DataFrame
    :param sample_rate: The data's sample rate in Hz.
    :type sample_rate: int or float
    :param lowpass_hz: Cutoff frequency (Hz) for Butterworth lowpass filter.
        Defaults to 20 Hz. Pass None or False to disable filtering.
    :type lowpass_hz: int or False, optional
    :param calibrate_gravity: Whether to perform gravity calibration using the
        method of van Hees et al. 2014. Defaults to True.
    :type calibrate_gravity: bool, optional
    :param detect_nonwear: Whether to detect and flag non-wear periods (long
        stationary periods). Defaults to True.
    :type detect_nonwear: bool, optional
    :param resample_hz: Target frequency (Hz) to resample the signal. If
        "uniform", uses the provided sample_rate to fix sampling errors. Pass
        None or False to disable. Defaults to "uniform".
    :type resample_hz: "uniform" or int or float or bool, optional
    :param start_first_complete_minute: Whether to start data from the first
        complete minute (with 1 second tolerance). Useful for aligning data to
        minute boundaries. Defaults to False.
    :type start_first_complete_minute: bool, optional
    :param calibrate_gravity_kwargs: Additional keyword arguments for the
        calibrate_gravity function (e.g., {'stdtol_min': 0.01}). Defaults to None.
    :type calibrate_gravity_kwargs: dict, optional
    :param flag_nonwear_kwargs: Additional keyword arguments for the flag_nonwear
        function (e.g., {'patience': '60m'}). Defaults to None.
    :type flag_nonwear_kwargs: dict, optional
    :param verbose: Print progress messages. Defaults to True.
    :type verbose: bool, optional
    :return: A tuple (data, info) where:

        - **data** (pandas.DataFrame): Processed acceleration time-series
        - **info** (dict): Processing metadata. See GLOSSARY.md for field descriptions.
    :rtype: tuple(pandas.DataFrame, dict)

    Examples
    --------
    Process a CSV file:

    >>> import pandas as pd
    >>> import actipy
    >>> data = pd.read_csv("custom_data.csv", parse_dates=['time'], index_col='time')
    >>> data, info = actipy.process(data, sample_rate=100)

    Apply specific processing steps:

    >>> data, info = actipy.process(
    ...     data,
    ...     sample_rate=100,
    ...     lowpass_hz=20,
    ...     calibrate_gravity=True,
    ...     detect_nonwear=False,
    ...     resample_hz=50
    ... )

    See Also
    --------
    read_device : Read and process device file in one step
    actipy.processing : Individual processing functions for more control

    Notes
    -----
    - Input data must have DateTimeIndex and columns 'x', 'y', 'z'
    - Optional columns 'temperature', 'light' are preserved if present
    - Processing is memory-efficient using chunked operations
    """

    _validate_resample_frequency(resample_hz)
    timer = Timer(verbose)

    info: Info = {}

    if lowpass_hz not in (None, False):
        timer.start("Lowpass filter...")
        data, info_lowpass = P.lowpass(data, sample_rate, cast(float, lowpass_hz))
        info.update(info_lowpass)
        timer.stop()

    if calibrate_gravity:
        timer.start("Gravity calibration...")
        calib_kwargs = calibrate_gravity_kwargs or {}
        data, info_calib = P.calibrate_gravity(data, **calib_kwargs)
        info.update(info_calib)
        timer.stop()

    if detect_nonwear:
        timer.start("Nonwear detection...")
        nonwear_kwargs = flag_nonwear_kwargs or {}
        data, info_nonwear = P.flag_nonwear(data, **nonwear_kwargs)
        info.update(info_nonwear)
        timer.stop()

    if resample_hz is not None and resample_hz is not False:
        timer.start("Resampling...")
        if resample_hz == 'uniform' or resample_hz is True:
            data, info_resample = P.resample(data, sample_rate, start_first_complete_minute=start_first_complete_minute)
        else:
            data, info_resample = P.resample(data, cast(float, resample_hz), start_first_complete_minute=start_first_complete_minute)
        info.update(info_resample)
        timer.stop()

    return data, info


def _read_device(input_file: str, verbose: bool = True) -> Tuple[pd.DataFrame, Info]:
    """ Internal function that interfaces with the Java parser to read the
    device file. Returns parsed data as a pandas dataframe, and a dict with
    general info.
    """

    # Use a separate reader if the file is from a Matrix device
    if matrix_reader.is_matrix_bin_file(input_file):
        return _read_device_matrix(input_file, verbose)

    try:

        timer = Timer(verbose)

        # Temporary directory for intermediate parser files.
        tmpdir = tempfile.mkdtemp()

        info: Info = {}
        info['Filename'] = input_file
        info['Filesize(MB)'] = round(os.path.getsize(input_file) / (1024 * 1024), 1)

        if input_file.lower().endswith((".gz", ".zip")):
            timer.start("Decompressing...")
            input_file = decompr(input_file, target_dir=tmpdir)
            timer.stop()

        # Device info
        info_device = get_device_info(input_file)
        info.update(info_device)

        # Parsing. Main action happens here.
        timer.start("Reading file...")
        info_java = java_read_device(input_file, tmpdir, verbose)
        info.update(info_java)
        timer.stop()

        timer.start("Converting to dataframe...")
        # NOTE: To reduce memory usage, we open the file as mmap and use
        # np.asarray to define the dict for the dataframe construction. Note
        # that the dataframe uses copy=True because we need to remove all
        # references to the mmap object so that the temporary directory can be
        # deleted later (another way is to use np.array and copy=False).
        data_mmap = np.load(os.path.join(tmpdir, "data.npy"), mmap_mode='r')
        data = pd.DataFrame({c: np.asarray(data_mmap[c]) for c in data_mmap.dtype.names}, copy=True)
        data.set_index('time', inplace=True)
        del data_mmap  # delete this so that the temporary directory can be deleted
        timer.stop()

        return data, info

    finally:

        # Cleanup, delete temporary directory
        try:
            # NOTE: For the tmpdir to be deleted, all references to the mmap
            # object must have been deleted. This includes data_mmap, but also
            # indirect references like the dataframe (copy=False) or arrays
            # created with np.asarray.
            shutil.rmtree(tmpdir)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")


def _read_device_matrix(input_file: str, verbose: bool = True) -> Tuple[pd.DataFrame, Info]:
    """ Internal function that reads a Matrix device file specifically. Returns
    parsed data as a pandas dataframe, and a dict with general info.
    """
    try:

        timer = Timer(verbose)

        # Temporary directory for intermediate conversion files.
        tmpdir = tempfile.mkdtemp()

        info: Info = {}
        info['Filename'] = input_file
        info['Filesize(MB)'] = round(os.path.getsize(input_file) / (1024 * 1024), 1)
        info['Device'] = 'Matrix'
        info['DeviceID'] = 'Matrix'

        # Decompress file if it is compressed
        if input_file.lower().endswith((".gz", ".zip")):
            timer.start("Decompressing...")
            input_file = decompr(input_file, target_dir=tmpdir)
            timer.stop()

        # Parsed data will be extracted to a CSV file
        output_file = os.path.join(tmpdir, "data.csv")

        # Parsing. Main action happens here.
        print("Reading file...")
        matrix_reader.bin2csv(input_file, output_file)
        print("Done!")

        timer.start("Converting to dataframe...")
        data = pd.read_csv(
            output_file,
            index_col='time',
            dtype={
                'x': 'f4', 'y': 'f4', 'z': 'f4',
                'gyro_x': 'f4', 'gyro_y': 'f4', 'gyro_z': 'f4',
                'body_surface_temperature': 'f4', 'ambient_temperature': 'f4',
                'heart_rate_raw': 'f4', 'heart_rate': 'f4'
            },
        )
        data.index = pd.to_datetime(data.index, unit='ms')
        # TODO:
        info['ReadOK'] = 1
        info['ReadErrors'] = 0
        info['SampleRate'] = infer_sample_rate(data.index)
        timer.stop()

        return data, info

    finally:

        # Cleanup, delete temporary directory
        try:
            # Remove the temporary directory and its intermediate CSV file.
            shutil.rmtree(tmpdir)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")



def java_read_device(input_file: str, output_dir: str, verbose: bool = True) -> Info:
    """Call the Java reader for a supported device file."""

    if input_file.lower().endswith('.cwa'):
        java_reader = 'AxivityReader'

    elif input_file.lower().endswith('.gt3x'):
        java_reader = 'ActigraphReader'

    elif input_file.lower().endswith('.bin'):
        java_reader = 'GENEActivReader'

    else:
        raise ValueError(f"Unknown file extension: {input_file}")

    command: List[str] = [
        "java",
        "-XX:ParallelGCThreads=1",
        "-cp", str(pathlib.Path(__file__).parent),
        java_reader,
        "-i", input_file,
        "-o", output_dir
    ]
    if verbose:
        command.append("-v")
    subprocess.run(command, check=True)

    # Load info.txt file. Each line is a key:value pair.
    with open(os.path.join(output_dir, "info.txt"), 'r') as f:
        info: Info = dict(line.split(':') for line in f.read().splitlines())

    info['ReadOK'] = int(info['ReadOK'])
    info['ReadErrors'] = int(info['ReadErrors'])
    info['SampleRate'] = float(info['SampleRate'])

    return info


def decompr(input_file: str, target_dir: str) -> str:
    """Decompress a supported archive into ``target_dir``."""

    # The Java readers accept decompressed .gz and .zip inputs.
    filename = os.path.basename(input_file)
    uncompr_filename = os.path.splitext(filename)[0]
    newfile = os.path.join(target_dir, uncompr_filename)

    if input_file.lower().endswith(".gz"):
        with gzip.open(input_file, 'rb') as fin:
            with open(newfile, 'wb') as fout:
                shutil.copyfileobj(fin, fout)

    elif input_file.lower().endswith(".zip"):
        with zipfile.ZipFile(input_file, 'r') as f:
            f.extractall(target_dir)

    return newfile


def get_device_info(input_file: str) -> Info:
    """Return the device type and serial number for ``input_file``."""

    info: Info = {}

    if input_file.lower().endswith('.bin'):
        info['Device'] = 'GENEActiv'
        info['DeviceID'] = get_genea_id(input_file)

    elif input_file.lower().endswith('.cwa'):
        info['Device'] = 'Axivity'
        info['DeviceID'] = get_axivity_id(input_file)

    elif input_file.lower().endswith('.gt3x'):
        info['Device'] = 'Actigraph'
        info['DeviceID'] = get_gt3x_id(input_file)

    elif input_file.lower().endswith('.csv'):
        info['Device'] = 'unknown (.csv)'
        info['DeviceID'] = 'unknown (.csv)'

    else:
        raise ValueError(f"Unknown file extension: {input_file}")

    return info


def get_axivity_id(cwafile: str) -> Union[int, str]:
    """Return the serial number embedded in an Axivity file."""

    f: IO[bytes]
    if cwafile.lower().endswith('.gz'):
        f = cast(IO[bytes], gzip.open(cwafile, 'rb'))
    else:
        f = open(cwafile, 'rb')

    header = f.read(2)
    if header == b'MD':
        struct.unpack('H', f.read(2))  # block size
        struct.unpack('B', f.read(1))  # perform-clear flag
        device_id: Union[int, str] = struct.unpack('H', f.read(2))[0]
    else:
        print(f"Could not find device id for {cwafile}")
        device_id = "unknown"

    f.close()

    return device_id


def get_genea_id(binfile: str) -> str:
    """Return the serial number embedded in a GENEActiv file."""

    assert binfile.lower().endswith(".bin"), f"Cannot get device id for {binfile}"

    with open(binfile, 'r') as f:
        next(f)  # Skip the device identity line.
        device_id = next(f).split(':')[1].rstrip()  # Device Unique Serial Code field.

    return device_id


def get_gt3x_id(gt3xfile: str) -> Optional[str]:
    """Return the serial number embedded in an ActiGraph archive."""

    # ActiGraph files are ZIP containers with an info.txt member.
    assert gt3xfile.lower().endswith(".gt3x") and zipfile.is_zipfile(gt3xfile), f"Cannot get device id for {gt3xfile}"

    with zipfile.ZipFile(gt3xfile, 'r') as z:
        if 'info.txt' in z.namelist():
            with z.open('info.txt', 'r') as info_file:
                for line in info_file:
                    newline = line.decode("utf-8-sig").strip()
                    if newline.startswith("Serial Number:"):
                        return newline.split(":", 1)[1].strip()
            return None
        else:
            print("Could not find info.txt file")
            return "unknown"


def fix_nonincr_time(data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """ Fix if time non-increasing (rarely occurs) """
    errs = (data.index.to_series().diff() <= pd.Timedelta(0)).sum()
    if errs > 0:
        print("Found non-increasing data timestamps. Fixing...")
        data = data[data.index.to_series()
                    .cummax()
                    .diff()
                    .fillna(pd.Timedelta(1))
                    > pd.Timedelta(0)]
    return data, cast(int, errs)


def infer_sample_rate(t: pd.DatetimeIndex) -> float:
    """ Like pd.infer_freq but more forgiving """
    tdiff = t.to_series().diff()
    q1, q3 = tdiff.quantile([0.25, 0.75])
    tdiff = tdiff[(q1 <= tdiff) & (tdiff <= q3)]
    dt = tdiff.mean()
    sample_rate = pd.Timedelta('1s') / pd.Timedelta(dt)
    return cast(float, sample_rate)


class Timer:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.msg: Optional[str] = None

    def start(self, msg: str = "Starting timer...") -> None:
        assert self.start_time is None, "Timer is running. Use .stop() to stop it"
        self.start_time = time.perf_counter()
        self.msg = msg
        if self.verbose:
            print(msg, end="\r")

    def stop(self) -> None:
        assert self.start_time is not None, "Timer is not running. Use .start() to start it"
        elapsed_time = time.perf_counter() - self.start_time
        if self.verbose:
            print(f"{self.msg} Done! ({elapsed_time:0.2f}s)")
        self.start_time = None
        self.msg = None
