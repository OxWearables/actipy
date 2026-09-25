#!/usr/bin/env python3
"""Build real-device parser windows and independent numerical oracles.

The full source recordings are intentionally not committed. This script selects
complete native records from their beginning, middle, and end, then decodes the
expected values without calling actipy or its Java readers.
"""

import argparse
import functools
import gzip
import hashlib
import io
import json
import math
import shutil
import struct
import zipfile
from array import array
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "tests" / "data" / "parser-fixtures"
DOTNET_UNIX_EPOCH_TICKS = 621355968000000000
WINDOW_SECONDS = 10 * 60
SOURCE_FILENAMES = {
    "actigraph_v1": "sample-actigraph.gt3x",
    "axivity": "sample-axivity.cwa.gz",
    "geneactiv": "sample-geneactiv.bin.gz",
}
ACTIGRAPH_V2_FILENAME = "sample-actigraph-leap.gt3x"
WINDOW_SUFFIXES = {"start": "", "middle": "-middle", "end": "-end"}

with (OUTPUT_DIR / "manifest.json").open(encoding="utf-8") as stream:
    COMMITTED_MANIFEST = json.load(stream)


@functools.lru_cache(maxsize=None)
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip_member(archive, name, payload):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            write_zip_member(archive, name, payload)


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
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, values in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, values, allow_pickle=False)
            write_zip_member(archive, f"{name}.npy", payload.getvalue())
    return len(arrays["time"]), list(columns)


def window_starts(item_count, window_size, alignment=1):
    if item_count < window_size:
        raise ValueError(
            f"Source has {item_count} items, fewer than window size {window_size}"
        )
    latest_start = item_count - window_size
    starts = {
        "start": 0,
        "middle": ((latest_start // 2) // alignment) * alignment,
        "end": (latest_start // alignment) * alignment,
    }
    if len(set(starts.values())) != len(starts):
        raise ValueError("Source is too short for distinct start/middle/end windows")
    return starts


def update_metadata(payload, changes):
    has_bom = payload.startswith(b"\xef\xbb\xbf")
    text = payload.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    trailing_newline = text.endswith(("\r\n", "\n"))
    updated = []
    found = set()
    for line in text.splitlines():
        key = line.split(":", 1)[0]
        if key in changes:
            line = f"{key}: {changes[key]}"
            found.add(key)
        updated.append(line)
    missing = set(changes) - found
    if missing:
        raise ValueError(f"Metadata fields not found: {sorted(missing)}")
    result = newline.join(updated) + (newline if trailing_newline else "")
    return (b"\xef\xbb\xbf" if has_bom else b"") + result.encode("utf-8")


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


def index_activity2_packets(payload):
    activity_offsets = array("I")
    position = 0
    while position < len(payload):
        if len(payload) - position < 8:
            raise ValueError("Actigraph V2 source ended inside a packet header")
        separator, record_type, _, size = struct.unpack_from(
            "<BBIH", payload, position
        )
        packet_end = position + 9 + size
        if packet_end > len(payload):
            raise ValueError("Actigraph V2 source ended inside a packet")
        if separator != 0x1E:
            raise ValueError(f"Invalid packet separator at byte {position}")
        if record_type == 26 and size > 1:
            activity_offsets.append(position)
        position = packet_end
    return activity_offsets


def split_packets(payload):
    packets = []
    position = 0
    while position < len(payload):
        header = payload[position:position + 8]
        if len(header) != 8:
            raise ValueError("Actigraph V2 selection ends inside a packet header")
        _, _, _, size = struct.unpack("<BBIH", header)
        packet_end = position + 9 + size
        packet = payload[position:packet_end]
        if len(packet) != 9 + size:
            raise ValueError("Actigraph V2 selection ends inside a packet")
        packets.append(packet)
        position = packet_end
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
        fixture, expected, source, *, rows, fields, sample_rate,
        device, device_id, selection):
    return {
        "fixture": fixture.name,
        "expected": expected.name,
        "source": source.name,
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


def build_actigraph_v1(source):
    with zipfile.ZipFile(source) as archive:
        metadata = archive.read("info.txt")
        all_activity = archive.read("activity.bin")
    source_metadata = parse_metadata(metadata)
    sample_rate = float(source_metadata["Sample Rate"])
    start_ticks = int(source_metadata["Start Date"])
    pair_count = int(sample_rate * WINDOW_SECONDS) // 2
    total_pairs = len(all_activity) // 9
    pair_alignment = int(sample_rate) // math.gcd(int(sample_rate), 2)
    starts = window_starts(total_pairs, pair_count, pair_alignment)
    entries = []
    for label, start_pair in starts.items():
        suffix = WINDOW_SUFFIXES[label]
        fixture = OUTPUT_DIR / f"actigraph-v1{suffix}.gt3x"
        expected = OUTPUT_DIR / f"actigraph-v1{suffix}-expected.npz"
        start_sample = start_pair * 2
        window_start_ticks = (
            start_ticks + int(start_sample / sample_rate) * 10_000_000
        )
        window_stop_ticks = window_start_ticks + WINDOW_SECONDS * 10_000_000
        window_metadata = update_metadata(
            metadata,
            {
                "Start Date": window_start_ticks,
                "Stop Date": window_stop_ticks,
            },
        )
        activity = all_activity[start_pair * 9:(start_pair + pair_count) * 9]
        write_zip(
            fixture,
            (("info.txt", window_metadata),
             ("activity.bin", activity),
             ("lux.bin", b"")),
        )
        time, columns, selected_rate, device_id = decode_actigraph_v1(
            window_metadata, activity
        )
        rows, fields = save_expected(expected, time, columns)
        entries.append(manifest_entry(
            fixture,
            expected,
            source,
            rows=rows,
            fields=fields,
            sample_rate=selected_rate,
            device="Actigraph",
            device_id=device_id,
            selection=(
                f"Ten-minute {label} window of complete nine-byte sample pairs."
            ),
        ))
    return entries


def build_actigraph_v2(source):
    with zipfile.ZipFile(source) as archive:
        metadata = archive.read("info.txt")
        log = archive.read("log.bin")
    source_metadata = parse_metadata(metadata)
    sample_rate = float(source_metadata["Sample Rate"])
    activity_offsets = index_activity2_packets(log)
    first_payload_size = struct.unpack_from(
        "<H", log, activity_offsets[0] + 6
    )[0]
    samples_per_packet = first_payload_size // 6
    window_packets = math.ceil(
        WINDOW_SECONDS * sample_rate / samples_per_packet
    )
    starts = window_starts(len(activity_offsets), window_packets)
    entries = []
    for label, activity_start in starts.items():
        suffix = WINDOW_SUFFIXES[label]
        fixture = OUTPUT_DIR / f"actigraph-leap-v2{suffix}.gt3x"
        expected = OUTPUT_DIR / f"actigraph-leap-v2{suffix}-expected.npz"
        byte_start = 0 if label == "start" else activity_offsets[activity_start]
        next_activity = activity_start + window_packets
        byte_end = (
            activity_offsets[next_activity]
            if next_activity < len(activity_offsets)
            else len(log)
        )
        packets = split_packets(log[byte_start:byte_end])
        write_zip(
            fixture,
            (("info.txt", metadata), ("log.bin", b"".join(packets))),
        )
        time, columns, selected_rate, device_id = decode_actigraph_v2(
            metadata, packets
        )
        rows, fields = save_expected(expected, time, columns)
        entries.append(manifest_entry(
            fixture,
            expected,
            source,
            rows=rows,
            fields=fields,
            sample_rate=selected_rate,
            device="Actigraph",
            device_id=device_id,
            selection=(
                f"Ten-minute {label} window of complete V2 packets, including "
                "interleaved non-activity records."
            ),
        ))
    return entries


def retain_actigraph_v2():
    entries = [
        item for item in COMMITTED_MANIFEST["fixtures"]
        if item["fixture"].startswith("actigraph-leap-v2")
    ]
    if not entries:
        raise ValueError("Committed manifest has no ActiGraph V2 fixtures")
    committed_dir = PROJECT_ROOT / "tests" / "data" / "parser-fixtures"
    if OUTPUT_DIR != committed_dir:
        for entry in entries:
            for key in ("fixture", "expected"):
                shutil.copy2(committed_dir / entry[key], OUTPUT_DIR / entry[key])
    return entries


def build_axivity(source):
    with source.open("rb") as compressed:
        compressed.seek(-4, 2)
        uncompressed_size = struct.unpack("<I", compressed.read(4))[0]
    total_blocks = uncompressed_size // 512
    data_block_count = total_blocks - 2

    with gzip.open(source, "rb") as stream:
        prefix = stream.read(3 * 512)
        if prefix[:2] != b"MD" or prefix[512:514] != b"\xff\xff":
            raise ValueError("Unexpected Axivity metadata/reserved block layout")
        first_data_block = prefix[2 * 512:]
        if first_data_block[:2] != b"AX":
            raise ValueError("Could not find first Axivity data block")
        rate_code = first_data_block[24]
        sample_rate = 3200.0 / (1 << (15 - (rate_code & 15)))
        samples_per_block = struct.unpack_from("<H", first_data_block, 28)[0]
        window_blocks = math.ceil(
            WINDOW_SECONDS * sample_rate / samples_per_block
        )
        starts = window_starts(data_block_count, window_blocks)
        selected_blocks = {
            "start": first_data_block + stream.read((window_blocks - 1) * 512)
        }
        for label in ("middle", "end"):
            stream.seek((2 + starts[label]) * 512)
            selected_blocks[label] = stream.read(window_blocks * 512)

    header = prefix[:2 * 512]
    entries = []
    for label in starts:
        suffix = WINDOW_SUFFIXES[label]
        fixture = OUTPUT_DIR / f"axivity-ax3{suffix}.cwa"
        expected = OUTPUT_DIR / f"axivity-ax3{suffix}-expected.npz"
        data_blocks = selected_blocks[label]
        if len(data_blocks) != window_blocks * 512:
            raise ValueError(f"Incomplete Axivity {label} window")
        payload = header + data_blocks
        fixture.write_bytes(payload)
        time, columns, selected_rate, device_id = decode_axivity(payload)
        rows, fields = save_expected(expected, time, columns)
        entries.append(manifest_entry(
            fixture,
            expected,
            source,
            rows=rows,
            fields=fields,
            sample_rate=selected_rate,
            device="Axivity",
            device_id=device_id,
            selection=(
                "Metadata and reserved blocks followed by a ten-minute "
                f"{label} window of complete packed AX3 blocks."
            ),
        ))
    return entries


def build_geneactiv(source):
    with gzip.open(source, "rb") as stream:
        header = [next(stream).rstrip(b"\n\r") for _ in range(59)]
        page_count_index = next(
            index for index, line in enumerate(header)
            if line.startswith(b"Number of Pages:")
        )
        total_pages = int(header[page_count_index].split(b":", 1)[1])
        first_page = [next(stream).rstrip(b"\n\r") for _ in range(10)]
        sample_rate = float(first_page[8].split(b":", 1)[1])
        samples_per_page = len(first_page[9]) // 12
        window_pages = math.ceil(
            WINDOW_SECONDS * sample_rate / samples_per_page
        )
        starts = window_starts(total_pages, window_pages)
        page_ranges = {
            label: range(start, start + window_pages)
            for label, start in starts.items()
        }
        selected_pages = {label: [] for label in starts}
        selected_pages["start"].extend(first_page)

        for page_index in range(1, total_pages):
            labels = [
                label for label, page_range in page_ranges.items()
                if page_index in page_range
            ]
            if labels:
                page = [next(stream).rstrip(b"\n\r") for _ in range(10)]
                for label in labels:
                    selected_pages[label].extend(page)
            else:
                for _ in range(10):
                    next(stream)

    entries = []
    for label, pages in selected_pages.items():
        if len(pages) != window_pages * 10:
            raise ValueError(f"Incomplete GENEActiv {label} window")
        suffix = WINDOW_SUFFIXES[label]
        fixture = OUTPUT_DIR / f"geneactiv{suffix}.bin"
        expected = OUTPUT_DIR / f"geneactiv{suffix}-expected.npz"
        window_header = list(header)
        window_header[page_count_index] = (
            f"Number of Pages:{window_pages}".encode("ascii")
        )
        payload = b"\n".join(window_header + pages) + b"\n"
        fixture.write_bytes(payload)
        time, columns, selected_rate, device_id = decode_geneactiv(payload)
        rows, fields = save_expected(expected, time, columns)
        entries.append(manifest_entry(
            fixture,
            expected,
            source,
            rows=rows,
            fields=fields,
            sample_rate=selected_rate,
            device="GENEActiv",
            device_id=device_id,
            selection=(
                f"Original header and a ten-minute {label} window of complete "
                f"pages; page count changed to {window_pages}."
            ),
        ))
    return entries


def find_source_dir(requested):
    candidates = [requested] if requested else [PROJECT_ROOT.parent, PROJECT_ROOT / "data"]
    for candidate in candidates:
        if candidate and all((candidate / name).is_file()
                             for name in SOURCE_FILENAMES.values()):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates if path)
    required = ", ".join(SOURCE_FILENAMES.values())
    raise FileNotFoundError(
        f"Could not find parser corpus in {searched}; required files: {required}"
    )


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for reduced fixtures, goldens, and manifest.",
    )
    parser.add_argument(
        "--source-dir", type=Path,
        help="Directory containing the three full real-device recordings.",
    )
    args = parser.parse_args()
    source_dir = find_source_dir(args.source_dir)
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fixtures = []
    fixtures.extend(build_actigraph_v1(
        source_dir / SOURCE_FILENAMES["actigraph_v1"]
    ))
    actigraph_v2_source = source_dir / ACTIGRAPH_V2_FILENAME
    if actigraph_v2_source.is_file():
        fixtures.extend(build_actigraph_v2(actigraph_v2_source))
    else:
        fixtures.extend(retain_actigraph_v2())
    fixtures.extend(build_axivity(
        source_dir / SOURCE_FILENAMES["axivity"]
    ))
    fixtures.extend(build_geneactiv(
        source_dir / SOURCE_FILENAMES["geneactiv"]
    ))
    manifest = {
        "format_version": 2,
        "window_seconds": WINDOW_SECONDS,
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
