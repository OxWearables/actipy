import argparse
import actipy

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    parser.add_argument('--lowpass_hz', default=None, type=float)
    parser.add_argument('--calibrate_gravity', action='store_true')
    parser.add_argument('--detect_nonwear', action='store_true')
    parser.add_argument('--resample_hz', default=None, type=float)
    args = parser.parse_args()

    data, info = actipy.read_device(
        args.datafile,
        lowpass_hz=args.lowpass_hz,
        calibrate_gravity=args.calibrate_gravity,
        detect_nonwear=args.detect_nonwear,
        resample_hz=args.resample_hz,
        verbose=True
    )

    print(data.head())

    # Pretty print info
    for k, v in info.items():
        print(f"{k:25s}: {v}")



if __name__ == '__main__':
    main()
