#!/usr/bin/env python3
"""Build small real-device parser fixtures and independent numerical oracles.

The source recordings live in ``data/`` and are intentionally not committed.
This script selects complete native records from them and decodes the expected
values without calling actipy or its Java readers.
"""

import argparse
import gzip
import hashlib
import json
import math
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parents[2]
SOURCE_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "data" / "parser-fixtures"
DOTNET_UNIX_EPOCH_TICKS = 621355968000000000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)


def parse_metadata(payload):
    metadata = {}
    for line in payload.decode("utf-8-sig").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def signed_12_bit(value):
    return value - 4096 if value >= 2048 else value


def java_round(value):
    return math.floor(value + 0.5)


def save_expected(path, time, columns):
    arrays = {"time": np.asarray(time, dtype=np.int64)}
    arrays.update(
        (name, np.asarray(values, dtype=np.float32))
        for name, values in columns.items()
    )
    np.savez_compressed(path, **arrays)
    return len(arrays["time"]), list(columns)


def decode_actigraph_v1(metadata_payload, activity):
    metadata = parse_metadata(metadata_payload)
    sample_rate = float(metadata["Sample Rate"])
    serial = metadata["Serial Number"]
    if serial.startswith(("NEO", "CLE")):
        scale = 341.0
    elif serial.startswith("MOS"):
        scale = 256.0
    else:
        scale = float(metadata["Acceleration Scale"])

    start_ticks = int(metadata["Start Date"])
    start_millis = (start_ticks - DOTNET_UNIX_EPOCH_TICKS) // 10_000
    samples = []
    for position in range(0, len(activity), 9):
        values = activity[position:position + 9]
        if len(values) != 9:
            raise ValueError("Actigraph V1 selection ends inside a sample pair")
        y1 = signed_12_bit((values[0] << 4) | (values[1] >> 4))
        x1 = signed_12_bit(((values[1] & 0x0F) << 8) | values[2])
        z1 = signed_12_bit((values[3] << 4) | (values[4] >> 4))
        y2 = signed_12_bit(((values[4] & 0x0F) << 8) | values[5])
        x2 = signed_12_bit((values[6] << 4) | (values[7] >> 4))
        z2 = signed_12_bit(((values[7] & 0x0F) << 8) | values[8])
        samples.extend(((x1, y1, z1), (x2, y2, z2)))

    time = [
        (start_millis + java_round(1000.0 * index / sample_rate)) * 1_000_000
        for index in range(len(samples))
    ]
    columns = {
        axis: [sample[index] / scale for sample in samples]
        for index, axis in enumerate(("x", "y", "z"))
    }
    return time, columns, sample_rate, serial


def read_packets(stream, count):
    packets = []
    for _ in range(count):
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError("Actigraph V2 source ended inside a packet header")
        _, _, _, size = struct.unpack("<BBIH", header)
        packet = header + stream.read(size + 1)
        if len(packet) != 9 + size:
            raise ValueError("Actigraph V2 source ended inside a packet")
        packets.append(packet)
    return packets


def decode_actigraph_v2(metadata_payload, packets):
    metadata = parse_metadata(metadata_payload)
    sample_rate = float(metadata["Sample Rate"])
    scale = float(metadata["Acceleration Scale"])
    serial = metadata["Serial Number"]
    time = []
    samples = []

    for packet in packets:
        separator, record_type, timestamp, size = struct.unpack("<BBIH", packet[:8])
        payload = packet[8:8 + size]
        expected_checksum = (~xor_bytes(packet[:8] + payload)) & 0xFF
        if separator != 0x1E or packet[-1] != expected_checksum:
            raise ValueError("Invalid Actigraph V2 packet in source selection")
        if record_type != 26 or size <= 1:
            continue
        if size % 6:
            raise ValueError("Actigraph V2 activity payload is not whole XYZ samples")
        for sample_index, (x_raw, y_raw, z_raw) in enumerate(
                struct.iter_unpack("<hhh", payload)):
            time.append(
                (timestamp * 1000
                 + java_round(1000.0 * sample_index / sample_rate))
                * 1_000_000
            )
            samples.append(tuple(
                java_round(raw / scale * 1000.0) / 1000.0
                for raw in (x_raw, y_raw, z_raw)
            ))

    columns = {
        axis: [sample[index] for sample in samples]
        for index, axis in enumerate(("x", "y", "z"))
    }
    return time, columns, sample_rate, serial


def xor_bytes(payload):
    value = 0
    for byte in payload:
        value ^= byte
    return value


def cwa_timestamp(value):
    date = datetime(
        2000 + ((value >> 26) & 0x3F),
        (value >> 22) & 0x0F,
        (value >> 17) & 0x1F,
        (value >> 12) & 0x1F,
        (value >> 6) & 0x3F,
        value & 0x3F,
        tzinfo=timezone.utc,
    )
    return int(date.timestamp())


def signed_10_bit(value):
    return value - 1024 if value >= 512 else value


def decode_axivity(payload):
    blocks = [payload[position:position + 512]
              for position in range(0, len(payload), 512)]
    if any(len(block) != 512 for block in blocks):
        raise ValueError("Axivity selection ends inside a block")

    time = []
    columns = {name: [] for name in ("x", "y", "z", "temperature", "light")}
    sample_rate = None
    last_block_time = 0.0
    for block in blocks:
        if block[:2] != b"AX":
            continue
        if sum(struct.unpack("<256h", block)) & 0xFFFF:
            raise ValueError("Invalid Axivity checksum in source selection")

        timestamp = cwa_timestamp(struct.unpack_from("<I", block, 14)[0])
        raw_light = struct.unpack_from("<H", block, 18)[0]
        raw_temperature = struct.unpack_from("<H", block, 20)[0]
        rate_code = block[24]
        axes_format = block[25]
        timestamp_offset = struct.unpack_from("<h", block, 26)[0]
        sample_count = struct.unpack_from("<H", block, 28)[0]
        num_axes, packing = axes_format >> 4, axes_format & 0x0F
        if num_axes != 3 or packing != 0:
            raise ValueError("Selected Axivity blocks are not packed AX3 data")

        sample_rate = np.float32(3200.0 / (1 << (15 - (rate_code & 15))))
        offset_start = np.float32(-timestamp_offset / sample_rate)
        whole_offset = math.floor(float(offset_start))
        timestamp += whole_offset
        offset_start = np.float32(offset_start - whole_offset)
        block_start = timestamp + float(offset_start)
        duration = np.float32(np.float32(sample_count) / sample_rate)
        block_end = block_start + float(duration)
        if last_block_time and block_start - last_block_time < 1.0:
            block_start = last_block_time
        last_block_time = block_end

        light = np.float32(10 ** ((raw_light & 0x3FF) / 341.0))
        temperature = np.float32(
            ((raw_temperature & 0x3FF) * 150.0 - 20500.0) / 1000.0
        )
        accel_unit = 1 << (8 + ((raw_light >> 13) & 0x07))
        for sample_index in range(sample_count):
            packed = struct.unpack_from("<I", block, 30 + 4 * sample_index)[0]
            exponent = packed >> 30
            raw = (
                signed_10_bit(packed & 0x3FF) << exponent,
                signed_10_bit((packed >> 10) & 0x3FF) << exponent,
                signed_10_bit((packed >> 20) & 0x3FF) << exponent,
            )
            millis = int(
                (block_start
                 + sample_index * (block_end - block_start) / sample_count)
                * 1000.0
            )
            time.append(millis * 1_000_000)
            for axis, value in zip(("x", "y", "z"), raw):
                columns[axis].append(np.float32(value) / np.float32(accel_unit))
            columns["temperature"].append(temperature)
            columns["light"].append(light)

    device_id = struct.unpack_from("<H", blocks[0], 5)[0]
    return time, columns, float(sample_rate), device_id


def decode_geneactiv(payload):
    lines = payload.decode("ascii").splitlines()
    gains = [float(lines[index].split(":", 1)[1].strip())
             for index in (47, 49, 51)]
    offsets = [int(lines[index].split(":", 1)[1].strip())
               for index in (48, 50, 52)]
    device_id = lines[1].split(":", 1)[1].strip()
    time = []
    columns = {name: [] for name in ("x", "y", "z", "temperature")}
    sample_rate = None

    for position in range(59, len(lines), 10):
        block = lines[position:position + 10]
        if len(block) != 10:
            raise ValueError("GENEActiv selection ends inside a page")
        block_time = datetime.strptime(
            block[3].split("Time:", 1)[1], "%Y-%m-%d %H:%M:%S:%f"
        ).replace(tzinfo=timezone.utc)
        block_millis = int(block_time.timestamp() * 1000)
        temperature = np.float32(block[5].split(":", 1)[1])
        sample_rate = float(block[8].split(":", 1)[1])
        encoded = block[9]
        if len(encoded) % 12:
            raise ValueError("GENEActiv page contains a partial sample")
        for sample_index in range(len(encoded) // 12):
            sample = encoded[sample_index * 12:(sample_index + 1) * 12]
            raw = [signed_12_bit(int(sample[index:index + 3], 16))
                   for index in (0, 3, 6)]
            millis = int(block_millis + sample_index * (1.0 / sample_rate) * 1000.0)
            time.append(millis * 1_000_000)
            for axis_index, axis in enumerate(("x", "y", "z")):
                value = (raw[axis_index] * 100.0 - offsets[axis_index]) / gains[axis_index]
                columns[axis].append(np.float32(value))
            columns["temperature"].append(temperature)

    return time, columns, sample_rate, device_id


def manifest_entry(
        fixture, expected, source, rows, fields, sample_rate,
        device, device_id, selection):
    return {
        "fixture": fixture.name,
        "expected": expected.name,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "fixture_sha256": sha256(fixture),
        "expected_sha256": sha256(expected),
        "source_sha256": sha256(source),
        "selection": selection,
        "rows": rows,
        "fields": ["time"] + fields,
        "sample_rate": sample_rate,
        "device": device,
        "device_id": device_id,
        "read_ok": 1,
        "read_errors": 0,
    }


def build_actigraph_v1():
    source = SOURCE_DIR / "sample-actigraph.gt3x"
    fixture = OUTPUT_DIR / "actigraph-v1.gt3x"
    expected = OUTPUT_DIR / "actigraph-v1-expected.npz"
    with zipfile.ZipFile(source) as archive:
        metadata = archive.read("info.txt")
        activity = archive.read("activity.bin")[:900]
    write_zip(fixture, (("info.txt", metadata), ("activity.bin", activity), ("lux.bin", b"")))
    time, columns, sample_rate, device_id = decode_actigraph_v1(metadata, activity)
    rows, fields = save_expected(expected, time, columns)
    return manifest_entry(
        fixture, expected, source, rows, fields, sample_rate,
        "Actigraph", device_id,
        "First 100 complete nine-byte sample pairs (200 samples).",
    )


def build_actigraph_v2():
    source = SOURCE_DIR / "sample-actigraph-leap.gt3x"
    fixture = OUTPUT_DIR / "actigraph-leap-v2.gt3x"
    expected = OUTPUT_DIR / "actigraph-leap-v2-expected.npz"
    with zipfile.ZipFile(source) as archive:
        metadata = archive.read("info.txt")
        with archive.open("log.bin") as stream:
            packets = read_packets(stream, 8)
    write_zip(fixture, (("info.txt", metadata), ("log.bin", b"".join(packets))))
    time, columns, sample_rate, device_id = decode_actigraph_v2(metadata, packets)
    rows, fields = save_expected(expected, time, columns)
    return manifest_entry(
        fixture, expected, source, rows, fields, sample_rate,
        "Actigraph", device_id,
        "First eight complete packets: one type 6, two type 2, and five type 26.",
    )


def build_axivity():
    source = SOURCE_DIR / "sample-axivity.cwa.gz"
    fixture = OUTPUT_DIR / "axivity-ax3.cwa"
    expected = OUTPUT_DIR / "axivity-ax3-expected.npz"
    with gzip.open(source, "rb") as stream:
        payload = stream.read(5 * 512)
    fixture.write_bytes(payload)
    time, columns, sample_rate, device_id = decode_axivity(payload)
    rows, fields = save_expected(expected, time, columns)
    return manifest_entry(
        fixture, expected, source, rows, fields, sample_rate,
        "Axivity", device_id,
        "Metadata and reserved blocks followed by three packed AX3 data blocks.",
    )


def build_geneactiv():
    source = SOURCE_DIR / "sample-geneactiv.bin.gz"
    fixture = OUTPUT_DIR / "geneactiv.bin"
    expected = OUTPUT_DIR / "geneactiv-expected.npz"
    with gzip.open(source, "rt", encoding="ascii") as stream:
        lines = [next(stream).rstrip("\n\r") for _ in range(89)]
    page_count_index = next(
        index for index, line in enumerate(lines[:59])
        if line.startswith("Number of Pages:")
    )
    lines[page_count_index] = "Number of Pages:3"
    payload = ("\n".join(lines) + "\n").encode("ascii")
    fixture.write_bytes(payload)
    time, columns, sample_rate, device_id = decode_geneactiv(payload)
    rows, fields = save_expected(expected, time, columns)
    return manifest_entry(
        fixture, expected, source, rows, fields, sample_rate,
        "GENEActiv", device_id,
        "Original 59-line header and first three complete pages; page count changed to 3.",
    )


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for reduced fixtures, goldens, and manifest.",
    )
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fixtures = [
        build_actigraph_v1(),
        build_actigraph_v2(),
        build_axivity(),
        build_geneactiv(),
    ]
    manifest = {
        "format_version": 1,
        "oracle": (
            "Expected arrays are decoded by the standalone format-specific "
            "implementations in tests/tools/build_real_parser_fixtures.py; "
            "actipy and the Java readers are not imported or executed."
        ),
        "fixtures": fixtures,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
