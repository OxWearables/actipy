import os
import time
import struct
import shutil
import tempfile
import atexit
import zipfile
import gzip
import pathlib
import numpy as np
import pandas as pd
import jpype

from actipy import processing as P


__all__ = ['read_device', 'process']


def read_device(input_file,
                lowpass_hz=20,
                calibrate_gravity=True,
                detect_nonwear=True,
                resample_hz='uniform',
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
    :param verbose: Verbosity, defaults to True.
    :type verbose: bool, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    data, info_read = _read_device(input_file, verbose)

    data, info_process = process(data, info_read['SampleRate'],
                                 lowpass_hz=lowpass_hz,
                                 calibrate_gravity=calibrate_gravity,
                                 detect_nonwear=detect_nonwear,
                                 resample_hz=resample_hz,
                                 verbose=verbose)

    info = {**info_read, **info_process}

    return data, info


def process(data, sample_rate,
            lowpass_hz=20,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz='uniform',
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

    # Used for calibration and nonwear detection
    # If needed, compute it once as it's expensive
    stationary_indicator = None
    if calibrate_gravity or detect_nonwear:
        timer.start("Getting stationary points...")
        stationary_indicator = P.get_stationary_indicator(data)
        timer.stop()

    if calibrate_gravity:
        timer.start("Gravity calibration...")
        data, info_calib = P.calibrate_gravity(data, stationary_indicator=stationary_indicator)
        info.update(info_calib)
        timer.stop()

    if detect_nonwear:
        timer.start("Nonwear detection...")
        data, info_nonwear = P.detect_nonwear(data, stationary_indicator=stationary_indicator)
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

    try:

        timer = Timer(verbose)

        # Temporary diretory to store internal runtime files
        tmpdir = tempfile.mkdtemp()
        # Temporary file to store parsed device data
        tmpout = os.path.join(tmpdir, "tmpout.npy")

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
        info_java = java_read_device(input_file, tmpout, verbose)
        info.update(info_java)
        timer.stop()

        timer.start("Converting to dataframe...")
        # Load parsed data to a pandas dataframe
        data = P.npy2df(np.load(tmpout, mmap_mode='r+')).copy()  # Note: needs copy or cleanup below will fail
        # Fix if time non-increasing (rarely occurs)
        data, nonincr_time_errs = fix_nonincr_time(data)
        # Update read errors. Non-increasing time errors scaled by sample rate
        info['ReadErrors'] += int(np.ceil(nonincr_time_errs / info['SampleRate']))
        timer.stop()

        # Start/end times, wear time, interrupts
        t = data.index.to_series()
        strftime = "%Y-%m-%d %H:%M:%S"
        info['StartTime'], info['EndTime'] = t[0].strftime(strftime), t[-1].strftime(strftime)
        info['NumTicks'] = len(data)
        wear_time, num_interrupts = P.get_wear_time(t)
        info['WearTime(days)'] = wear_time / (60 * 60 * 24)
        info['NumInterrupts'] = num_interrupts

        return data, info

    finally:

        # Cleanup, delete temporary directory
        try:
            shutil.rmtree(tmpdir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


def java_read_device(input_file, output_file, verbose):
    """ Core function that calls the Java method to read device data """

    setupJVM()

    if input_file.lower().endswith('.cwa'):
        info = jpype.JClass('AxivityReader').read(input_file, output_file, verbose)

    elif input_file.lower().endswith('.gt3x'):
        info = jpype.JClass('ActigraphReader').read(input_file, output_file, verbose)

    elif input_file.lower().endswith('.bin'):
        info = jpype.JClass('GENEActivReader').read(input_file, output_file, verbose)

    else:
        raise ValueError(f"Unknown file extension: {input_file}")

    # Convert the Java HashMap object to Python dictionary
    info = {str(k): str(info[k]) for k in info}
    info['ReadOK'] = int(info['ReadOK'])
    info['ReadErrors'] = int(info['ReadErrors'])
    info['SampleRate'] = float(info['SampleRate'])

    return info


def setupJVM():
    """ Start JVM. Shutdown at program exit """
    if not jpype.isJVMStarted():
        jpype.addClassPath(pathlib.Path(__file__).parent)
        jpype.startJVM(convertStrings=False)

        @atexit.register
        def shudownJVM():
            jpype.shutdownJVM()

    return


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
