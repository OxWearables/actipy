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
                regularize_sample_rate=False,
                calibrate_gravity=False,
                detect_nonwear=False,
                check_quality=False,
                verbose=True):

    info = {}

    # Basic info
    info['filename'] = input_file
    info['filesize(MB)'] = int(round(os.path.getsize(input_file) / (1024*1024), 0))
    info['args'] = {
        'regularize_sample_rate': regularize_sample_rate,
        'calibrate_gravity': calibrate_gravity,
        'detect_nonwear': detect_nonwear,
        'check_quality': check_quality,
        'verbose': verbose
    }

    data, info_read = _read_device(input_file, verbose)
    data = npy2df(data)

    info_resample = {}
    if regularize_sample_rate:
        data, info_resample = processing.regularize_sample_rate(data, info_read['sampleRate'])

    info_calib, info_nonwear = {}, {}
    if calibrate_gravity or detect_nonwear:
        stationary_indicator = processing.get_stationary_indicator(data)

        if calibrate_gravity:
            data, info_calib = processing.calibrate_gravity(data, stationary_indicator=stationary_indicator)

        if detect_nonwear:
            data, info_nonwear = processing.detect_nonwear(data, stationary_indicator=stationary_indicator)

    info.update({**info_read,
                 **info_resample,
                 **info_calib,
                 **info_nonwear})

    return data, info


def _read_device(input_file, verbose=True):

    before = time.time()

    info = {}

    # Setup
    setupJVM()
    # Create a temporary diretory to store intermediate results
    # This gets deleted at program exit
    tmpdir = make_tmpdir()
    tmpnpy = os.path.join(tmpdir, "tmp.npy")
    if verbose:
        print("Decompressing...", end="\r")
    input_file_decompr = check_and_decompr(input_file, target_dir=tmpdir)

    # Parsing
    if verbose:
        print("Reading file... ", end="\r")
    if input_file_decompr.lower().endswith('.cwa'):
        info['device'] = "Axivity"
        info_parse = jpype.JClass('AxivityReader').read(input_file_decompr, tmpnpy, verbose)

    elif input_file_decompr.lower().endswith('.gt3x'):
        info['device'] = "Actigraph"
        info_parse = jpype.JClass('ActigraphReader').read(input_file_decompr, tmpnpy, verbose)

    elif input_file_decompr.lower().endswith('.bin'):
        info['device'] = "GENEActiv"
        info_parse = jpype.JClass('GENEActivReader').read(input_file_decompr, tmpnpy, verbose)

    else:
        raise ValueError(f"Unknown file format {input_file}")

    # Device ID
    info['deviceID'] = get_device_id(input_file_decompr)

    # Convert the Java HashMap object to Python dictionary
    info_parse = {str(k): str(info_parse[k]) for k in info_parse}
    info_parse['readOK'] = int(info_parse['readOK'])
    info_parse['readErrors'] = int(info_parse['readErrors'])
    info_parse['sampleRate'] = float(info_parse['sampleRate'])

    info.update(info_parse)

    # Load result
    data = np.load(tmpnpy, mmap_mode='r')

    if verbose:
        print(f"Reading file... Done! ({time.time()-before:.2f}s)")

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


def check_and_decompr(filepath, target_dir):

    # Only .gz and .zip supported so far
    if filepath.lower().endswith((".gz", ".zip")):
        filename = os.path.basename(filepath)
        uncompr_filename = os.path.splitext(filename)[0]
        newpath = os.path.join(target_dir, uncompr_filename)

        if filepath.lower().endswith(".gz"):
            with gzip.open(filepath, 'rb') as fin:
                with open(newpath, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)

        elif filepath.lower().endswith(".zip"):
            with zipfile.ZipFile(filepath, 'r') as f:
                f.extractall(target_dir)

    else:
        newpath = filepath

    assert newpath.lower().endswith((".cwa", ".bin", ".gt3x")), f"Unknown file format {filepath}"

    return newpath


def make_tmpdir():
    """ Create temporary directory. Remove at program exit """
    tmpdir = tempfile.mkdtemp()

    @atexit.register
    def remove_tempdir():
        try:
            shutil.rmtree(tmpdir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    return tmpdir


def npy2df(data):
    before = time.time()
    print("Converting to pandas dataframe...", end=" ", flush=True)
    data = pd.DataFrame(data)
    # Set time
    data['time'] = data['time'].astype('datetime64[ms]')
    data = data.set_index('time')
    print(f"Done! ({time.time()-before:.2f}s)")

    return data


def get_device_id(inpfile):
    """ Get serial number of device """

    if inpfile.lower().endswith('.bin'):
        return get_genea_id(inpfile)
    elif inpfile.lower().endswith(('.cwa', '.cwa.gz')):
        return get_axivity_id(inpfile)
    elif inpfile.lower().endswith('.gt3x'):
        return get_gt3x_id(inpfile)
    elif inpfile.lower().endswith(('.csv', '.csv.gz')):
        return "unknown (.csv)"
    else:
        print(f"Could not find device id for {inpfile}")
        return "unknown"


def get_axivity_id(cwafile):
    """ Get serial number of Axivity device """

    if cwafile.lower().endswith('.gz'):
        f = gzip.open(cwafile,'rb')
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

    with open(binfile, 'r') as f: # 'Universal' newline mode
        next(f) # Device Identity
        device_id = next(f).split(':')[1].rstrip() # Device Unique Serial Code:011710

    return device_id


def get_gt3x_id(gt3xfile):
    """ Get serial number of Actigraph device """

    # Actigraph is actually a zip file?
    assert gt3xfile.lower().endswith(".gt3x") and zipfile.is_zipfile(gt3xfile), f"Cannot get device id for {gt3xfile}"

    with zipfile.ZipFile(gt3xfile, 'r') as z:
        contents = z.infolist()
        # print("\n".join(map(lambda x: str(x.filename).rjust(20, " ") + ", " + str(x.file_size), contents)))

        if 'info.txt' in map(lambda x: x.filename, contents):
            # print('info.txt found..')
            info_file = z.open('info.txt', 'r')
            # print info_file.read()
            for line in info_file:
                if line.startswith(b"Serial Number:"):
                    newline = line.decode("utf-8")
                    newline = newline.split("Serial Number: ")[1]
                    # print("Serial Number: "+newline)
                    return newline
        else:
            print("Could not find info.txt file")
            return "unknown"
