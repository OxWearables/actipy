import os
import logging
import struct
import binascii
import csv
import io
import gzip
import zipfile
import tarfile
from typing import BinaryIO
from tqdm.auto import tqdm

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="[%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger(__name__)

# Constants
REMARKS_SIZE = 512
FILE_SIGNATURE = b'MDTC'
FILE_HEADER_STRUCT = struct.Struct("<4sIHH")
PACKET_SIGNATURE = b'MDTCPACK'
PACKET_HEADER_STRUCT = struct.Struct('<8sIIIIIII')
ACC_STRUCT = struct.Struct('hhh')    # 6 bytes
GYRO_STRUCT = struct.Struct('hhh')   # 6 bytes
TEMP_STRUCT = struct.Struct('hh')    # 4 bytes
HR_STRUCT = struct.Struct('hh')      # 4 bytes
CSV_HEADER = [
    'time', 'x', 'y', 'z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'body_surface_temperature', 'ambient_temperature',
    'heart_rate_raw', 'heart_rate'
]


def bin2csv(bin_path: str, csv_path: str) -> None:
    """
    Converts a binary (.bin) data file from a Matrix wearable device into a
    Comma Separated Values (.csv) file.

    The function reads the binary file, parses its header and remark sections,
    and then processes each data packet. Each packet contains sensor readings
    for accelerometer, gyroscope, temperature (body surface and ambient), and
    heart rate.

    The binary file structure is expected to have:
    1. A remarks block (512 bytes).
    2. A file header (4sIHH format) containing a signature, number of packets,
       accelerometer range, and gyroscope range.
    3. A series of data packets, each starting with PACKET_SIGNATURE.

    Each data packet is parsed for its header (containing CRC, timestamps,
    and sample counts for different sensors) and payload. The payload data
    is then scaled and resampled.

    :param bin_path: Path to the input Matrix wearable '.bin' file.
    :type bin_path: str
    :param csv_path: Path where the output '.csv' file will be saved.
    :type csv_path: str
    :return: Implicitly returns None upon successful completion.
             The primary output is the generated CSV file.
    """

    metadata = extract_metadata(bin_path)
    remarks = metadata['remarks']
    num_packets = metadata['num_packets']
    acc_range = metadata['acc_range']
    gyro_range = metadata['gyro_range']
    LOGGER.debug(f"Remarks: {remarks}")
    LOGGER.debug(f"Number of packets: {num_packets}, "
                 f"Accelerometer range: {acc_range}, "
                 f"Gyroscope range: {gyro_range}")

    # Scale factors for accelerometer and gyroscope
    acc_scale_pos = acc_range / 0x7FFF
    acc_scale_neg = acc_range / 0x8000
    gyro_scale_pos = gyro_range / 0x7FFF
    gyro_scale_neg = gyro_range / 0x8000

    # Read the entire file into memory
    with open(bin_path, 'rb') as f:
        file_data = bytearray(f.read())

    # Locate all packet start offsets
    offsets = find_offets(file_data)
    if not offsets:
        raise ValueError("No packets found.")
    LOGGER.debug(f"Number of offsets: {len(offsets)}.")

    num_offsets = len(offsets)
    file_data_size = len(file_data)
    file_data = memoryview(file_data)
    previous_t1 = None

    with open(csv_path, 'w', encoding='utf-8_sig', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(CSV_HEADER)

        # Process each packet
        for packet_idx, packet_start in enumerate(tqdm(offsets)):
            packet_end = offsets[packet_idx + 1] if (packet_idx + 1) < num_offsets else file_data_size
            packet = file_data[packet_start:packet_end]

            # Minimum size check: header is 8s + I*7 = 8 + 28 = 36 bytes
            if len(packet) < 36:
                # too short to even contain header
                LOGGER.debug("Packet too short to contain valid header; skipping this packet.")
                continue

            try:
                (sig,
                 crc,
                 t0,
                 t1,
                 acc_count,
                 gyro_count,
                 temp_count,
                 hr_count) = PACKET_HEADER_STRUCT.unpack_from(packet, 0)
            except struct.error:
                LOGGER.debug("Failed to unpack packet header; skipping this packet.")
                continue

            # Verify that sig indeed starts with PACKET_SIGNATURE
            if sig != PACKET_SIGNATURE:
                LOGGER.debug(f"Invalid packet signature: {sig}. Expected {PACKET_SIGNATURE}. Skipping packet.")
                continue

            # Compute CRC32 over everything after the first 12 bytes (skip 8 bytes sig + 4 bytes crc field)
            computed_crc = binascii.crc32(packet[12:])
            if computed_crc != crc:
                LOGGER.debug(f"CRC mismatch: expected {crc}, got {computed_crc}. Skipping packet.")
                continue

            # Now pull out the payload bytes (after the full header)
            payload = packet[PACKET_HEADER_STRUCT.size:]

            # Compute how many aggregated samples we need to produce
            max_count = max(acc_count, gyro_count, temp_count, hr_count)
            if max_count <= 0:
                LOGGER.debug("No samples to process in this packet.")
                continue

            # Preallocate lists for each column; fill with '' by default
            acc_x_list = [''] * max_count
            acc_y_list = [''] * max_count
            acc_z_list = [''] * max_count
            gyro_x_list = [''] * max_count
            gyro_y_list = [''] * max_count
            gyro_z_list = [''] * max_count
            body_temp_list = [''] * max_count
            ambient_temp_list = [''] * max_count
            hr_raw_list = [''] * max_count
            hr_list = [''] * max_count

            # Fix gap between packets if needed
            if previous_t1 and t0 - previous_t1 == 1:
                LOGGER.debug(f"Gap detected between packets: {previous_t1} -> {t0}.")
                t0 = t0 - 1
            previous_t1 = t1

            # Unix times (in milliseconds)
            dt = (t1 - t0) / max_count
            time_list = [int(1000 * (t0 + i * dt)) for i in range(max_count)]

            # Now parse each raw sensor block one by one, assigning to the correct indices
            block_start = 0

            # ACC (acc_count samples)
            if acc_count > 0:
                step = max_count / acc_count
                block_end = block_start + acc_count * ACC_STRUCT.size
                indices = [int(i * step) for i in range(acc_count)]
                blocks = ACC_STRUCT.iter_unpack(payload[block_start:block_end])
                for (x, y, z), idx in zip(blocks, indices):
                    x = x * (acc_scale_pos if x > 0 else acc_scale_neg)
                    y = y * (acc_scale_pos if y > 0 else acc_scale_neg)
                    z = z * (acc_scale_pos if z > 0 else acc_scale_neg)
                    acc_x_list[idx] = f'{x:.6f}'
                    acc_y_list[idx] = f'{y:.6f}'
                    acc_z_list[idx] = f'{z:.6f}'
                block_start = block_end  # update slice_start to the end of the block

            # GYRO (gyro_count samples)
            if gyro_count > 0:
                step = max_count / gyro_count
                block_end = block_start + gyro_count * GYRO_STRUCT.size
                indices = [int(i * step) for i in range(gyro_count)]
                blocks = GYRO_STRUCT.iter_unpack(payload[block_start:block_end])
                for (x, y, z), idx in zip(blocks, indices):
                    x = x * (gyro_scale_pos if x > 0 else gyro_scale_neg)
                    y = y * (gyro_scale_pos if y > 0 else gyro_scale_neg)
                    z = z * (gyro_scale_pos if z > 0 else gyro_scale_neg)
                    gyro_x_list[idx] = f'{x:.3f}'
                    gyro_y_list[idx] = f'{y:.3f}'
                    gyro_z_list[idx] = f'{z:.3f}'
                block_start = block_end  # update slice_start to the end of the block

            # TEMPERATURE (temp_count samples)
            if temp_count > 0:
                step = max_count / temp_count
                block_end = block_start + temp_count * TEMP_STRUCT.size
                indices = [int(i * step) for i in range(temp_count)]
                blocks = TEMP_STRUCT.iter_unpack(payload[block_start:block_end])
                for (body_temp, ambient_temp), idx in zip(blocks, indices):
                    body_temp = body_temp * 0.1
                    ambient_temp = ambient_temp * 0.1
                    body_temp_list[idx] = body_temp
                    ambient_temp_list[idx] = ambient_temp
                block_start = block_end  # update slice_start to the end of the block

            # HEART RATE (hr_count samples)
            if hr_count > 0:
                step = max_count / hr_count
                block_end = block_start + hr_count * HR_STRUCT.size
                indices = [int(i * step) for i in range(hr_count)]
                blocks = HR_STRUCT.iter_unpack(payload[block_start:block_end])
                for (hr_raw_val, hr_val), idx in zip(blocks, indices):
                    hr_raw_list[idx] = hr_raw_val
                    hr_list[idx] = hr_val
                block_start = block_end  # update slice_start to the end of the block

            # Finally, write out each of the max_count rows
            for i in range(max_count):
                writer.writerow([
                    time_list[i],
                    acc_x_list[i], acc_y_list[i], acc_z_list[i],
                    gyro_x_list[i], gyro_y_list[i], gyro_z_list[i],
                    body_temp_list[i], ambient_temp_list[i],
                    hr_raw_list[i], hr_list[i],
                ])

            # --- end of processing this packet ---

    return metadata


def find_offets(file_data: bytearray) -> list:
    """
    Return a list of all indices where PACKET_SIGNATURE begins inside buffer.
    Uses a single-pass .find(…, start) to get O(N) overall.
    """
    offsets = []
    start = 0
    while True:
        idx = file_data.find(PACKET_SIGNATURE, start)
        if idx == -1:
            break
        offsets.append(idx)
        start = idx + 1
    return offsets


def extract_metadata(bin_path: str) -> dict:
    """
    Extracts metadata from a Matrix wearable binary file.

    This function serves as a high-level interface to read metadata.
    It handles opening the binary file (which might be plain, gzipped,
    zipped, or a tar.gz archive containing a .bin file) using `open_bin_file`,
    then passes the resulting file stream to `_extract_metadata` for parsing.
    The file stream is automatically closed after metadata extraction.

    :param bin_path: Path to the Matrix wearable binary file. This can be a
                     direct '.bin' file or a supported archive ('.gz', '.zip',
                     '.tar.gz', '.tgz') containing a single '.bin' file.
    :type bin_path: str
    :return: A dictionary containing the parsed metadata:
             - 'remarks' (str): The parsed remarks.
             - 'file_signature' (bytes): The 4-byte signature from the file header.
             - 'num_packets' (int): The number of data packets declared in the header.
             - 'acc_range' (int): The accelerometer range from the header.
             - 'gyro_range' (int): The gyroscope range from the header.
    :rtype: dict
    """
    f = open_bin_file(bin_path)
    try:
        return _extract_metadata(f)
    finally:
        f.close()


def _extract_metadata(f: BinaryIO) -> dict:
    """
    Reads and parses metadata from an open binary file stream of a Matrix wearable device.

    This function expects the file stream `f` to be positioned at the
    beginning of the Matrix binary data. It performs the following steps:
    1. Reads the first 512 bytes as the remarks block and parses it.
    2. Reads the subsequent bytes corresponding to the file header structure
       (4sIHH: a 4-byte signature, an unsigned int for the number of packets,
       and two unsigned shorts for accelerometer and gyroscope ranges).
    3. Validates the `file_signature` against the expected `file_signature`.

    :param f: An open binary file-like object (stream) pointing to the
              beginning of the Matrix '.bin' data.
    :type f: typing.BinaryIO
    :return: A dictionary containing the parsed metadata:
             - 'remarks' (str): The parsed remarks.
             - 'file_signature' (bytes): The 4-byte signature from the file header.
             - 'num_packets' (int): The number of data packets declared in the header.
             - 'acc_range' (int): The accelerometer range from the header.
             - 'gyro_range' (int): The gyroscope range from the header.
    :rtype: dict
    """
    # Read and parse 512‐byte REMARKS
    raw_remarks = read_bytes(f, REMARKS_SIZE)
    remarks = parse_remarks(raw_remarks)

    # Read and parse 4sIHH header
    raw_file_header = read_bytes(f, FILE_HEADER_STRUCT.size)
    file_signature, num_packets, acc_range, gyro_range = FILE_HEADER_STRUCT.unpack(raw_file_header)

    if file_signature != FILE_SIGNATURE:
        raise ValueError(
            f"Invalid file signature: {file_signature!r}. "
            f"Expected {FILE_SIGNATURE!r}. Is this a Matrix .bin?"
        )

    return {
        "remarks": remarks,
        "file_signature": file_signature,
        "num_packets": num_packets,
        "acc_range": acc_range,
        "gyro_range": gyro_range,
    }


def read_bytes(f, num_bytes):
    """
    Read exactly num_bytes from file f; if fewer bytes are available,
    return what we have and let caller detect truncation.
    """
    data = f.read(num_bytes)
    if len(data) < num_bytes:
        raise EOFError(f'Expected {num_bytes} bytes, got {len(data)}')
    return data


def parse_remarks(raw_bytes: bytes) -> str:
    """
    Take a 512-length bytes object, decode as UTF-8 (ignore errors),
    and strip at the first null byte if present.
    """
    decoded = raw_bytes.decode('utf-8', 'ignore')
    if '\0' in decoded:
        decoded = decoded.split('\0', 1)[0]
    return decoded


def is_matrix_bin_file(bin_path: str) -> bool:
    """
    Checks if the given file is a valid Matrix wearable binary file.

    This function attempts to extract metadata from the file. If metadata
    extraction is successful, the file is considered a valid Matrix binary file.
    It catches common errors that might occur if the file is not a valid
    Matrix binary file or a supported archive type (e.g., `ValueError` for
    incorrect header, `zipfile.BadZipFile` for corrupted ZIP archives,
    `gzip.BadGzipFile` for corrupted GZIP files).

    :param bin_path: Path to the file to check.
    :type bin_path: str
    :return: True if the file is a valid Matrix wearable binary file (or a
             supported archive containing one), False otherwise.
    :rtype: bool
    """
    try:
        _ = extract_metadata(bin_path)
    except (ValueError, zipfile.BadZipFile, gzip.BadGzipFile) as e:
        return False
    return True


def open_bin_file(path: str) -> BinaryIO:
    """
    Opens a Matrix wearable binary file, handling potential compression or archiving.

    This function can open:
    - Plain '.bin' files.
    - gzip-compressed '.gz' files (containing a single '.bin' file).
    - ZIP archives '.zip' (containing a single '.bin' file).
    - Tarred gzip archives '.tar.gz' or '.tgz' (containing a single '.bin' file).

    It returns a binary file-like object (stream) pointing to the
    uncompressed '.bin' data. For archives, it extracts the '.bin' file.
    If the archive contains multiple '.bin' files or no '.bin' files,
    it raises a ValueError.

    :param path: The file path to the Matrix wearable data.
                 This can be a direct '.bin' file or a compressed/archived
                 file ('.gz', '.zip', '.tar.gz', '.tgz') containing a '.bin' file.
    :type path: str
    :return: A binary I/O stream (file-like object) to the uncompressed '.bin' data.
    :rtype: typing.BinaryIO
    """
    _, ext = os.path.splitext(path.lower())

    # Handle .tar.gz / .tgz
    if path.lower().endswith(".tar.gz") or path.lower().endswith(".tgz"):
        gz_stream = gzip.open(path, "rb")
        tar = tarfile.open(fileobj=gz_stream, mode="r:*")
        members = [m for m in tar.getmembers() if m.name.lower().endswith(".bin")]
        if not members:
            raise ValueError(f"No ‘.bin’ inside tar.gz {path!r}")
        if len(members) > 1:
            raise ValueError(
                f"Multiple .bin files in {path!r}: {', '.join(m.name for m in members)}"
            )
        extracted = tar.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"Cannot extract {members[0].name!r} from {path!r}")
        data = extracted.read()
        return io.BytesIO(data)

    # Handle .gz (not tar)
    if ext == ".gz":
        return gzip.open(path, "rb")

    # Handle .zip
    if ext == ".zip":
        zf = zipfile.ZipFile(path, "r")
        bin_names = [n for n in zf.namelist() if n.lower().endswith(".bin")]
        if not bin_names:
            raise ValueError(f"No .bin in ZIP {path!r}")
        if len(bin_names) > 1:
            raise ValueError(f"Multiple .bin in {path!r}: {bin_names}")
        # Option A: stream only the first 516 bytes (preferred if file very large):
        return zf.open(bin_names[0], "r")
        # # Option B: read fully into RAM if you need random access:
        # data = zf.read(bin_names[0])
        # return io.BytesIO(data)

    # Plain .bin (or “.dat”/“.raw” if you prefer)
    if ext == ".bin" or ext in ("", ".dat", ".raw"):
        return open(path, "rb")

    raise ValueError(f"Unrecognized extension: {path!r}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert Matrix wearable binary files to CSV format.')
    parser.add_argument('input_bin', type=str, help='Path to the input binary file')
    parser.add_argument('output_csv', type=str, help='Path to the output CSV file')
    parser.add_argument('--log', type=str, default=os.getenv('LOG_LEVEL', 'INFO').upper(), help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity (can be used multiple times)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress all output except errors')
    args = parser.parse_args()

    # Determine logging level based on parsed arguments
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = max(logging.DEBUG, logging.WARNING - args.verbose * 10)
    else:
        log_level = args.log.upper()
    LOGGER.setLevel(log_level)

    # Run the conversion
    try:
        bin2csv(args.input_bin, args.output_csv)
        logging.info("Conversion completed successfully.")
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        raise
