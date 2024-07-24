import time
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import os
import json
# from tqdm import tqdm

from actipy import read_device

"""
How to run the script:

```bash
python src/actipy/read_cwa.py data/test.bin 

python src/actipy/read_cwa.py data/test.bin -o data/prepared/ -r 30 -g -f 20 -w -c x y z -q
```
"""


def main():
    parser = argparse.ArgumentParser(
        description="A tool to read and extract data from an Axivity (.cwa) device, and save it to a .csv file",
        add_help=True,
    )
    parser.add_argument("filepath", help="Enter file to be processed")
    parser.add_argument("--outdir", "-o", help="Enter folder location to save output files", default="outputs/")
    parser.add_argument("--resample-hz", "-r", help="Resample rate for output data.", type=str, default=None)
    parser.add_argument("--lowpass-hz", "-f", help="Frequency of low pass filter.", type=str, default=None)
    parser.add_argument("--detect-nonwear", "-w", help="Detect non-wear.", action="store_true")
    parser.add_argument("--calibrate-gravity", "-g", help="Calibrate gravity.", action="store_true")
    parser.add_argument("--output-cols", "-c", help="Restrict output columns to those listed (excluding time index column). Output all columns if falsy.", type=str, nargs="+", default=None)
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output.")

    args = parser.parse_args()

    verbose = not args.quiet
    lowpass_hz = validate_lowpass_hz(args.lowpass_hz)
    resample_hz = validate_resample_hz(args.resample_hz)

    data, info = read_device(
        args.filepath,
        lowpass_hz=lowpass_hz,
        calibrate_gravity=args.calibrate_gravity,
        detect_nonwear=args.detect_nonwear,
        resample_hz=resample_hz,
        verbose=verbose,
    )

    output_cols = validate_output_cols(args.output_cols, data) or list(data.columns)
    data = data[output_cols]

    # Output paths
    basename = resolve_path(args.filepath)[1]
    outdir = Path(args.outdir) / basename
    outdir.mkdir(parents=True, exist_ok=True)

    csv_file = outdir / f"{basename}.csv.gz"
    if verbose:
        print("Saving dataframe to disk...", end="\r")
    before = time.perf_counter()
    data.to_csv(csv_file, index=True)
    elapsed_time = time.perf_counter() - before
    if verbose:
        print(f"Saving dataframe to disk... Done! ({elapsed_time:0.2f}s)")
        print(f"Dataframe saved to: {os.path.abspath(csv_file)}")

    info_file = outdir / f"{basename}-Info.json"
    with open(info_file, "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=4, cls=NpEncoder)
    if verbose:
        print(f"Info file saved to: {os.path.abspath(info_file)}")


def validate_resample_hz(resample_hz):
    if resample_hz is None or resample_hz.lower() in ["none", ""]:
        return None
    if resample_hz.lower() in ["true"]:
        return True
    if resample_hz.lower() in ["false"]:
        return False
    try:
        resample_hz = int(resample_hz)
    except ValueError:
        raise ValueError("Sample rate must be a number, 'None', 'True', or 'False'.")
    return resample_hz


def validate_lowpass_hz(lowpass_hz):
    if lowpass_hz is None or lowpass_hz.lower() in ["none", ""]:
        return None
    if lowpass_hz.lower() in ["false"]:
        return False
    try:
        lowpass_hz = int(lowpass_hz)
    except ValueError:
        raise ValueError("Lowpass hz must be a number, 'None', or 'False'.")
    return lowpass_hz


def validate_output_cols(output_cols, data: pd.DataFrame):
    if output_cols is None or output_cols == []:
        return None
    if len(output_cols) == 1:
        if output_cols[0].lower() in ["none", "", "false"]:
            return None

    for elem in output_cols:
        if elem not in data.columns:
            raise ValueError(
                f"Column {elem} is not a column in the extracted data: {list(data.columns)}."
            )

    return output_cols


def df_to_csv(
    df: pd.DataFrame, filename: str, progress_desc: str = "", verbose: bool = False
):
    if verbose:
        chunks = np.array_split(df.index, 1000)

        pbar = tqdm(
            total=len(chunks),
            bar_format="{desc}|{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]",
            desc=progress_desc,
        )

        for chunk, subset in enumerate(chunks):
            if chunk == 0:
                df.loc[subset].to_csv(filename, mode="w")
            else:
                df.loc[subset].to_csv(filename, header=None, mode="a")

            pbar.update(1)

        pbar.close()
    else:
        df.to_csv(filename, index=True)


def resolve_path(path):
    """ Return parent folder, file name and file extension """
    p = Path(path)
    extension = p.suffixes[0]
    filename = p.name.rsplit(extension)[0]
    dirname = p.parent
    return dirname, filename, extension


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isnull(obj):  # handles pandas NAType
            return np.nan
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    main()
