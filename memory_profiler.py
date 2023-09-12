import os
import datetime
import argparse
import memray
import subprocess

import actipy
from actipy.reader import Timer
from actipy import processing as P



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    parser.add_argument('--lowpass_hz', default=None, type=float)
    parser.add_argument('--calibrate_gravity', action='store_true')
    parser.add_argument('--detect_nonwear', action='store_true')
    parser.add_argument('--resample_hz', default=None, type=float)
    args = parser.parse_args()

    # Directory to store the profiler report
    memray_dir = f'memory_profiler_report/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(memray_dir, exist_ok=True)

    timer = Timer()

    with memray.Tracker(f'{memray_dir}/read_device.bin'):
        data, info = actipy.read_device(
            args.datafile,
            lowpass_hz=None,
            calibrate_gravity=False,
            detect_nonwear=False,
            resample_hz=None,
            verbose=True
        )

    if args.lowpass_hz is not None:
        timer.start('Lowpass filtering...')
        with memray.Tracker(f'{memray_dir}/lowpass.bin'):
            data, info_lowpass = P.lowpass(data, info['SampleRate'], cutoff_rate=args.lowpass_hz)
        timer.stop()
        info.update(info_lowpass)

    if args.calibrate_gravity:
        timer.start('Calibrating gravity...')
        with memray.Tracker(f'{memray_dir}/calibrate_gravity.bin'):
            data, info_calibrate = P.calibrate_gravity(data, calib_cube=0, calib_min_samples=1)
        timer.stop()
        info.update(info_calibrate)

    if args.detect_nonwear:
        timer.start('Detecting nonwear...')
        with memray.Tracker(f'{memray_dir}/detect_nonwear.bin'):
            data, info_nonwear = P.detect_nonwear(data, patience='1m')
        timer.stop()
        info.update(info_nonwear)

    if args.resample_hz is not None:
        timer.start('Resampling...')
        with memray.Tracker(f'{memray_dir}/resample.bin'):
            data, info_resample = P.resample(data, args.resample_hz)
        timer.stop()
        info.update(info_resample)

    # Run flamegraph on each output file
    for fname in os.listdir(memray_dir):
        if fname.endswith('.bin'):
            subprocess.run(['python3', '-m', 'memray', 'flamegraph', f'{memray_dir}/{fname}'], check=True)

    print(data)

    # Pretty print info
    for k, v in info.items():
        print(f"{k:25s}: {v}")



if __name__ == '__main__':
    main()
