import os
import time
import struct
import shutil
import tempfile
import zipfile
import gzip
import pathlib
import numpy as np
import pandas as pd
import subprocess

from actipy import matrix_reader
from actipy import processing as P


__all__ = ['read_device', 'process']


def read_device(input_file,
                lowpass_hz=20,
                calibrate_gravity=True,
                detect_nonwear=True,
                resample_hz='uniform',
                start_time=None,
                end_time=None,
                skipdays=0,
                cutdays=0,
                calibrate_gravity_kwargs=None,
                flag_nonwear_kwargs=None,
                verbose=True):
    """
    Read and process accelerometer device file. Returns a pandas.DataFrame with
    the processed data and a dict with processing info.

    :param input_file: Path to accelerometer file.
    :type input_file: str
    :param lowpass_hz: Cutoff (Hz) for low-pass filter. Defaults to 20. Pass
        None or False to disable.
    :type lowpass_hz: int, optional
    :param calibrate_gravity: Whether to perform gravity calibration. Defaults to True.
    :type calibrate_gravity: bool, optional
    :param detect_nonwear: Whether to perform non-wear detection. Defaults to True.
    :type detect_nonwear: bool, optional
    :param resample_hz: Target frequency (Hz) to resample the signal. If
        "uniform", use the implied frequency (use this option to fix any device
        sampling errors). Pass None to disable. Defaults to "uniform".
    :type resample_hz: str or int, optional
    :param start_time: Start time to read data. Pass None to read from the
        beginning. Defaults to None.
    :type start_time: str or datetime, optional
    :param end_time: End time to read data. Pass None to read until the end.
        Defaults to None.
    :type end_time: str or datetime, optional
    :param skipdays: Number of days to skip from the beginning. Defaults to 0.
    :type skipdays: int, optional
    :param cutdays: Number of days to cut from the end. Defaults to 0.
    :type cutdays: int, optional
    :param calibrate_gravity_kwargs: Additional keyword arguments to pass to calibrate_gravity function. Defaults to None.
    :type calibrate_gravity_kwargs: dict, optional
    :param flag_nonwear_kwargs: Additional keyword arguments to pass to flag_nonwear function. Defaults to None.
    :type flag_nonwear_kwargs: dict, optional
    :param verbose: Verbosity, defaults to True.
    :type verbose: bool, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    timer = Timer(verbose)

    data, info = _read_device(input_file, verbose)

    # Filter data by start/end time, if specified
    if start_time is not None:
        data = data.loc[start_time:]
    if end_time is not None:
        data = data.loc[:end_time]

    # Skip/cut days, if specified
    if skipdays > 0:
        data = data.loc[data.index[0] + pd.Timedelta(days=skipdays):]
    if cutdays > 0:
        data = data.loc[:data.index[-1] - pd.Timedelta(days=cutdays)]

    # data, info_process = process(data, info_read['SampleRate'],
    #                              lowpass_hz=lowpass_hz,
    #                              calibrate_gravity=calibrate_gravity,
    #                              detect_nonwear=detect_nonwear,
    #                              resample_hz=resample_hz,
    #                              verbose=verbose)

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
        data, info_lowpass = P.lowpass(data, info['SampleRate'], lowpass_hz)
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

    if resample_hz not in (None, False):
        timer.start("Resampling...")
        if resample_hz in ('uniform', True):
            data, info_resample = P.resample(data, info['SampleRate'])
        else:
            data, info_resample = P.resample(data, resample_hz)
        info.update(info_resample)
        timer.stop()

    return data, info


def process(data, sample_rate,
            lowpass_hz=20,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz='uniform',
            calibrate_gravity_kwargs=None,
            flag_nonwear_kwargs=None,
            verbose=True):
    """
    Process a pandas.DataFrame of acceleration time-series. Returns a
    pandas.DataFrame with the processed data and a dict with processing info.

    :param data: A pandas.DataFrame of acceleration time-series. It must contain
        at least columns `x,y,z` and the index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param sample_rate: The data's sample rate (Hz).
    :type sample_rate: int or float
    :param lowpass_hz: Cutoff (Hz) for low-pass filter. Defaults to 20. Pass
        None or False to disable.
    :type lowpass_hz: int, optional
    :param calibrate_gravity: Whether to perform gravity calibration. Defaults to True.
    :type calibrate_gravity: bool, optional
    :param detect_nonwear: Whether to perform non-wear detection. Defaults to True.
    :type detect_nonwear: bool, optional
    :param resample_hz: Target frequency (Hz) to resample the signal. If
        "uniform", use the implied frequency (use this option to fix any device
        sampling errors). Pass None to disable. Defaults to "uniform".
    :type resample_hz: str or int, optional
    :param calibrate_gravity_kwargs: Additional keyword arguments to pass to calibrate_gravity function. Defaults to None.
    :type calibrate_gravity_kwargs: dict, optional
    :param flag_nonwear_kwargs: Additional keyword arguments to pass to flag_nonwear function. Defaults to None.
    :type flag_nonwear_kwargs: dict, optional
    :param verbose: Verbosity, defaults to True.
    :type verbose: bool, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    timer = Timer(verbose)

    info = {}

    if lowpass_hz not in (None, False):
        timer.start("Lowpass filter...")
        data, info_lowpass = P.lowpass(data, sample_rate, lowpass_hz)
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

    if resample_hz not in (None, False):
        timer.start("Resampling...")
        if resample_hz in ('uniform', True):
            data, info_resample = P.resample(data, sample_rate)
        else:
            data, info_resample = P.resample(data, resample_hz)
        info.update(info_resample)
        timer.stop()

    return data, info


def _read_device(input_file, verbose=True):
    """ Internal function that interfaces with the Java parser to read the
    device file. Returns parsed data as a pandas dataframe, and a dict with
    general info.
    """

    # Use a separate reader if the file is from a Matrix device
    if matrix_reader.is_matrix_bin_file(input_file):
        return _read_device_matrix(input_file, verbose)

    try:

        timer = Timer(verbose)

        # Temporary diretory to store internal runtime files
        tmpdir = tempfile.mkdtemp()

        info = {}
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


def _read_device_matrix(input_file, verbose=True):
    """ Internal function that reads a Matrix device file specifically. Returns
    parsed data as a pandas dataframe, and a dict with general info.
    """
    try:

        timer = Timer(verbose)

        # Temporary diretory to store internal runtime files
        tmpdir = tempfile.mkdtemp()

        info = {}
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
            # NOTE: For the tmpdir to be deleted, all references to the mmap
            # object must have been deleted. This includes data_mmap, but also
            # indirect references like the dataframe (copy=False) or arrays
            # created with np.asarray.
            shutil.rmtree(tmpdir)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")



def java_read_device(input_file, output_dir, verbose=True):
    """ Core function that calls the Java method to read device data """

    if input_file.lower().endswith('.cwa'):
        java_reader = 'AxivityReader'

    elif input_file.lower().endswith('.gt3x'):
        java_reader = 'ActigraphReader'

    elif input_file.lower().endswith('.bin'):
        java_reader = 'GENEActivReader'

    else:
        raise ValueError(f"Unknown file extension: {input_file}")

    command = [
        "java",
        "-XX:ParallelGCThreads=1",
        "-cp", pathlib.Path(__file__).parent,
        java_reader,
        "-i", input_file,
        "-o", output_dir
    ]
    if verbose:
        command.append("-v")
    subprocess.run(command, check=True)

    # Load info.txt file. Each line is a key:value pair.
    with open(os.path.join(output_dir, "info.txt"), 'r') as f:
        info = dict([line.split(':') for line in f.read().splitlines()])

    info['ReadOK'] = int(info['ReadOK'])
    info['ReadErrors'] = int(info['ReadErrors'])
    info['SampleRate'] = float(info['SampleRate'])

    return info


def decompr(input_file, target_dir):
    """ Decompress file to target_dir """

    # Only .gz and .zip supported so far
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


def get_device_info(input_file):
    """ Get serial number of device """

    info = {}

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


def get_axivity_id(cwafile):
    """ Get serial number of Axivity device """

    if cwafile.lower().endswith('.gz'):
        f = gzip.open(cwafile, 'rb')
    else:
        f = open(cwafile, 'rb')

    header = f.read(2)
    if header == b'MD':
        block_size = struct.unpack('H', f.read(2))[0]
        perform_clear = struct.unpack('B', f.read(1))[0]
        device_id = struct.unpack('H', f.read(2))[0]
    else:
        print(f"Could not find device id for {cwafile}")
        device_id = "unknown"

    f.close()

    return device_id


def get_genea_id(binfile):
    """ Get serial number of GENEActiv device """

    assert binfile.lower().endswith(".bin"), f"Cannot get device id for {binfile}"

    with open(binfile, 'r') as f:  # 'Universal' newline mode
        next(f)  # Device Identity
        device_id = next(f).split(':')[1].rstrip()  # Device Unique Serial Code:011710

    return device_id


def get_gt3x_id(gt3xfile):
    """ Get serial number of Actigraph device """

    # Actigraph is actually a zip file?
    assert gt3xfile.lower().endswith(".gt3x") and zipfile.is_zipfile(gt3xfile), f"Cannot get device id for {gt3xfile}"

    with zipfile.ZipFile(gt3xfile, 'r') as z:
        contents = z.infolist()

        if 'info.txt' in map(lambda x: x.filename, contents):
            info_file = z.open('info.txt', 'r')
            for line in info_file:
                if line.startswith(b"Serial Number:"):
                    newline = line.decode("utf-8")
                    newline = newline.split("Serial Number: ")[1]
                    return newline
        else:
            print("Could not find info.txt file")
            return "unknown"


def fix_nonincr_time(data):
    """ Fix if time non-increasing (rarely occurs) """
    errs = (data.index.to_series().diff() <= pd.Timedelta(0)).sum()
    if errs > 0:
        print("Found non-increasing data timestamps. Fixing...")
        data = data[data.index.to_series()
                    .cummax()
                    .diff()
                    .fillna(pd.Timedelta(1))
                    > pd.Timedelta(0)]
    return data, errs


def infer_sample_rate(t):
    """ Like pd.infer_freq but more forgiving """
    tdiff = t.to_series().diff()
    q1, q3 = tdiff.quantile([0.25, 0.75])
    tdiff = tdiff[(q1 <= tdiff) & (tdiff <= q3)]
    dt = tdiff.mean()
    sample_rate = pd.Timedelta('1s') / pd.Timedelta(dt)
    return sample_rate


class Timer:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.start_time = None
        self.msg = None

    def start(self, msg="Starting timer..."):
        assert self.start_time is None, "Timer is running. Use .stop() to stop it"
        self.start_time = time.perf_counter()
        self.msg = msg
        if self.verbose:
            print(msg, end="\r")

    def stop(self):
        assert self.start_time is not None, "Timer is not running. Use .start() to start it"
        elapsed_time = time.perf_counter() - self.start_time
        if self.verbose:
            print(f"{self.msg} Done! ({elapsed_time:0.2f}s)")
        self.start_time = None
        self.msg = None
