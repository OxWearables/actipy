import os
import time
import struct
import shutil
import tempfile
import atexit
import zipfile
import gzip
import numpy as np
import pandas as pd
import jpype

from pywear import processing


__all__ = ['read_device']


def read_device(input_file,
                resample_uniform=False,
                remove_noise=False,
                calibrate_gravity=False,
                detect_nonwear=False,
                verbose=True):
    """ Read and process device file. Returns a pandas.DataFrame with the
    processed data, and a dict with processing and general info. """

    info = {}

    # Basic info
    info['filename'] = input_file
    info['filesize(MB)'] = round(os.path.getsize(input_file) / (1024 * 1024), 1)

    data, info_read = _read_device(input_file, verbose)
    info.update(info_read)

    data, info_process = _process(data, info,
                                  resample_uniform=resample_uniform,
                                  remove_noise=remove_noise,
                                  calibrate_gravity=calibrate_gravity,
                                  detect_nonwear=detect_nonwear,
                                  verbose=verbose)
    info.update(info_process)

    return data, info


def _read_device(input_file, verbose=True):
    """ Internal function that interfaces with the Java parser to read the
    device file. Returns parsed data as a pandas dataframe, and a dict with
    general info.
    """

    try:

        timer = Timer(verbose)

        info = {}

        # Temporary diretory to store internal runtime files
        tmpdir = tempfile.mkdtemp()
        # Temporary file to store parsed device data
        tmpout = os.path.join(tmpdir, "tmpout.npy")

        if input_file.lower().endswith((".gz", ".zip")):
            timer.start("Decompressing...")
            input_file = decompr(input_file, target_dir=tmpdir)
            timer.stop()

        # Device info
        info_device = get_device_info(input_file)

        # Parsing. Main action happens here.
        timer.start("Reading file...")
        info_read = java_device_read(input_file, tmpout, verbose)
        timer.stop()

        info.update({**info_device, **info_read})

        # Load the parsed data to a pandas dataframe
        timer.start("Converting to dataframe...")
        data = npy2df(np.load(tmpout, mmap_mode='r'))
        timer.stop()

        return data, info

    finally:

        # Cleanup, delete temporary directory
        try:
            shutil.rmtree(tmpdir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


def java_device_read(input_file, output_file, verbose):
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
    info['readOK'] = int(info['readOK'])
    info['readErrors'] = int(info['readErrors'])
    info['sampleRate'] = float(info['sampleRate'])

    return info


def _process(data, info_data,
             resample_uniform=False,
             remove_noise=False,
             calibrate_gravity=False,
             detect_nonwear=False,
             verbose=False):
    """ Internal helper function to process data """

    timer = Timer(verbose)

    info = {}

    # Noise removal routine requires the data be uniformly sampled
    if remove_noise:
        resample_uniform = True

    if resample_uniform:
        timer.start("Resampling...")
        data, info_resample = processing.resample(data, info_data['sampleRate'])
        info.update(info_resample)
        timer.stop()

    if remove_noise:
        timer.start("Removing noise...")
        data, info_noise = processing.remove_noise(data, info_resample['resampleRate'],
                                                   resample_uniform=False)  # no need as already resampled
        info.update(info_noise)
        timer.stop()

    # Used for calibration and nonwear detection
    # If needed, compute it once as it's expensive
    stationary_indicator = None
    if calibrate_gravity or detect_nonwear:
        timer.start("Getting stationary points...")
        stationary_indicator = processing.get_stationary_indicator(data)
        timer.stop()

    if calibrate_gravity:
        timer.start("Gravity calibration...")
        data, info_calib = processing.calibrate_gravity(data, stationary_indicator=stationary_indicator)
        info.update(info_calib)
        timer.stop()

    if detect_nonwear:
        timer.start("Nonwear detection...")
        data, info_nonwear = processing.detect_nonwear(data, stationary_indicator=stationary_indicator)
        info.update(info_nonwear)
        timer.stop()

    return data, info


def setupJVM():
    """ Start JVM. Shutdown at program exit """
    if not jpype.isJVMStarted():
        jpype.addClassPath(os.path.join(os.environ['PYWEARPATH'], 'pywear/'))
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


def npy2df(data):
    """ Convert numpy array to pandas dataframe.
    Also parse time and set it as index. """

    data = pd.DataFrame(data)
    data['time'] = data['time'].astype('datetime64[ms]')
    data = data.set_index('time')

    return data


def get_device_info(input_file):
    """ Get serial number of device """

    info = {}

    if input_file.lower().endswith('.bin'):
        info['device'] = 'GENEActiv'
        info['deviceID'] = get_genea_id(input_file)

    elif input_file.lower().endswith('.cwa'):
        info['device'] = 'Axivity'
        info['deviceID'] = get_axivity_id(input_file)

    elif input_file.lower().endswith('.gt3x'):
        info['device'] = 'Actigraph'
        info['deviceID'] = get_gt3x_id(input_file)

    elif input_file.lower().endswith('.csv'):
        info['device'] = 'unknown (.csv)'
        info['deviceID'] = 'unknown (.csv)'

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
        # blockSize = struct.unpack('H', f.read(2))[0]
        # performClear = struct.unpack('B', f.read(1))[0]
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
