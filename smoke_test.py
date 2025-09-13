import argparse
import actipy

def main():

    parser = argparse.ArgumentParser(description='Process accelerometer data files')
    parser.add_argument('datafile', help='Path to accelerometer data file')
    parser.add_argument('--lowpass_hz', default=None, type=float,
                        help='Cutoff frequency (Hz) for low-pass filter')
    parser.add_argument('--calibrate_gravity', action='store_true',
                        help='Perform gravity calibration')
    parser.add_argument('--detect_nonwear', action='store_true',
                        help='Detect and flag non-wear periods')
    parser.add_argument('--resample_hz', default=None, type=float,
                        help='Target frequency (Hz) for resampling')
    parser.add_argument('--start_time', default=None, type=str,
                        help='Start time for data (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', default=None, type=str,
                        help='End time for data (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--skipdays', default=0, type=int,
                        help='Number of days to skip from beginning')
    parser.add_argument('--cutdays', default=0, type=int,
                        help='Number of days to cut from end')
    parser.add_argument('--start_first_complete_minute', action='store_true',
                        help='Start data from first complete minute (1 second tolerance)')
    args = parser.parse_args()

    data, info = actipy.read_device(
        args.datafile,
        lowpass_hz=args.lowpass_hz,
        calibrate_gravity=args.calibrate_gravity,
        detect_nonwear=args.detect_nonwear,
        resample_hz=args.resample_hz,
        start_time=args.start_time,
        end_time=args.end_time,
        skipdays=args.skipdays,
        cutdays=args.cutdays,
        start_first_complete_minute=args.start_first_complete_minute,
        verbose=True
    )

    print(data.head())

    # Pretty print info
    for k, v in info.items():
        print(f"{k:25s}: {v}")



if __name__ == '__main__':
    main()
