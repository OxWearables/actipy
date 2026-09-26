import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;


public class ActigraphReader {

    private static final int INVALID_GT3_FILE = 0;
    private static final int VALID_GT3_V1_FILE = 1;
    private static final int VALID_GT3_V2_FILE = 2;
    private static final int GT3_HEADER_SIZE = 8;
    private static final int GT3_SYNC_BYTE = 0x1E;
    private static final int PARAMETER_ID = 21;
    private static final int ACTIVITY_ID = 0;
    private static final int ACTIVITY2_ID = 26;
    private static final int INPUT_BUFFER_SIZE = 64 * 1024;
    private static final double MIN_ACCELERATION_SCALE = 16.0;
    private static final double MAX_ACCELERATION_SCALE = 32768.0;
    private static final double MIN_RAW_FULL_SCALE = 1024.0;
    private static final double MAX_RAW_FULL_SCALE = 32768.0;

    private static final class Metadata {
        double sampleRate = -1;
        double accelerationScale = -1;
        double accelerationMin = Double.NaN;
        double accelerationMax = Double.NaN;
        boolean accelerationScalePresent;
        long firstSampleTime = -1;
        String serialNumber = "";
    }

    public static void main(String[] args) {
        ReaderSupport.run(args, ActigraphReader::convert);
    }

    private static void convert(
            ReaderSupport.Options options,
            ReaderSupport.Result result) throws Exception {
        try (ZipFile zip = new ZipFile(options.inputFile)) {
            int version = getGT3XVersion(zip);
            if (version == INVALID_GT3_FILE) {
                throw new ReaderSupport.FormatException(
                        "File is not a supported V1 or V2 GT3X archive");
            }

            ZipEntry infoEntry = zip.getEntry("info.txt");
            ZipEntry activityEntry = version == VALID_GT3_V1_FILE
                    ? zip.getEntry("activity.bin")
                    : zip.getEntry("log.bin");
            Metadata metadata;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                    zip.getInputStream(infoEntry), StandardCharsets.UTF_8))) {
                metadata = readMetadata(reader, options.inputFile);
            }
            validateMetadata(metadata, version);
            result.sampleRate = metadata.sampleRate;

            try (InputStream activity = new BufferedInputStream(
                         zip.getInputStream(activityEntry), INPUT_BUFFER_SIZE);
                 NpyWriter writer = options.createWriter(NpyWriter.Layout.XYZ)) {
                if (version == VALID_GT3_V1_FILE) {
                    readV1(activity, metadata, writer, result);
                } else {
                    readV2(activity, metadata, writer, result);
                }
            }
        }
    }

    private static Metadata readMetadata(
            BufferedReader reader,
            String inputFile) throws IOException {
        Metadata metadata = new Metadata();
        String line;
        while ((line = reader.readLine()) != null) {
            String[] tokens = line.split(": ", 2);
            if (tokens.length != 2) {
                continue;
            }

            String key = tokens[0].trim();
            if (key.startsWith("\uFEFF")) {
                key = key.substring(1).trim();
            }
            String value = tokens[1].trim();

            if ("Sample Rate".equals(key)) {
                metadata.sampleRate = Double.parseDouble(value);
            } else if ("Start Date".equals(key)) {
                metadata.firstSampleTime = ticksToMilliseconds(Long.parseLong(value));
            } else if ("Acceleration Scale".equals(key)) {
                metadata.accelerationScale = Double.parseDouble(value);
                metadata.accelerationScalePresent = true;
            } else if ("Acceleration Min".equals(key)) {
                metadata.accelerationMin = Double.parseDouble(value);
            } else if ("Acceleration Max".equals(key)) {
                metadata.accelerationMax = Double.parseDouble(value);
            } else if ("Serial Number".equals(key)) {
                metadata.serialNumber = value;
            }
        }

        if (!isValidAccelerationScale(metadata.accelerationScale)
                || !isScaleConsistentWithRange(
                        metadata.accelerationScale,
                        metadata.accelerationMin,
                        metadata.accelerationMax)) {
            if (metadata.accelerationScalePresent) {
                System.err.println("Ignoring invalid Acceleration Scale in " + inputFile);
            }
            metadata.accelerationScale = accelerationScaleForSerial(metadata.serialNumber);
        }
        return metadata;
    }

    private static void validateMetadata(Metadata metadata, int version)
            throws ReaderSupport.FormatException {
        if (!Double.isFinite(metadata.sampleRate) || metadata.sampleRate <= 0) {
            throw new ReaderSupport.FormatException(
                    "GT3X metadata must contain a positive finite Sample Rate");
        }
        if (version == VALID_GT3_V1_FILE) {
            if (metadata.firstSampleTime < 0) {
                throw new ReaderSupport.FormatException(
                        "V1 GT3X metadata must contain Start Date");
            }
            if (!isValidAccelerationScale(metadata.accelerationScale)) {
                throw new ReaderSupport.FormatException(
                        "V1 GT3X metadata must contain a valid acceleration scale");
            }
        }
    }

    private static void readV1(
            InputStream input,
            Metadata metadata,
            NpyWriter writer,
            ReaderSupport.Result result)
            throws IOException, ReaderSupport.FormatException {
        byte[] inputBuffer = new byte[INPUT_BUFFER_SIZE];
        double[] samples = new double[6];
        long sampleIndex = 0;
        int bufferedBytes = 0;
        try {
            while (true) {
                int count = input.read(
                        inputBuffer,
                        bufferedBytes,
                        inputBuffer.length - bufferedBytes);
                if (count == -1) {
                    if (bufferedBytes == 5) {
                        decodeFirstPackedSample(
                                inputBuffer, 0, metadata.accelerationScale, samples, 0);
                        writeV1Sample(writer, metadata, sampleIndex, samples, 0);
                        sampleIndex++;
                    } else if (bufferedBytes != 0) {
                        throw new EOFException("Unexpected end of V1 packed activity");
                    }
                    break;
                }
                if (count == 0) {
                    continue;
                }
                bufferedBytes += count;

                int pairedBytes = bufferedBytes - bufferedBytes % 9;
                for (int offset = 0; offset < pairedBytes; offset += 9) {
                    decodePackedPair(
                            inputBuffer, offset, metadata.accelerationScale, samples);
                    writeV1Sample(writer, metadata, sampleIndex, samples, 0);
                    sampleIndex++;
                    writeV1Sample(writer, metadata, sampleIndex, samples, 3);
                    sampleIndex++;
                }

                bufferedBytes -= pairedBytes;
                if (bufferedBytes > 0) {
                    System.arraycopy(
                            inputBuffer, pairedBytes, inputBuffer, 0, bufferedBytes);
                }
            }
        } catch (EOFException error) {
            if (sampleIndex == 0) {
                throw error;
            }
            result.recordRecoverableError(
                    "Stopping at truncated GT3X data: " + error.getMessage());
        }
    }

    private static void writeV1Sample(
            NpyWriter writer,
            Metadata metadata,
            long sampleIndex,
            double[] samples,
            int offset) throws IOException {
        long timeMillis = metadata.firstSampleTime
                + Math.round(1000d * sampleIndex / metadata.sampleRate);
        writer.write(
                TimeUnit.MILLISECONDS.toNanos(timeMillis),
                (float) samples[offset],
                (float) samples[offset + 1],
                (float) samples[offset + 2]);
    }

    private static void readV2(
            InputStream input,
            Metadata metadata,
            NpyWriter writer,
            ReaderSupport.Result result)
            throws IOException, ReaderSupport.FormatException {
        byte[] header = new byte[GT3_HEADER_SIZE];
        byte[] payload = new byte[0];
        double[] packedSamples = new double[6];
        double accelerationScale = metadata.accelerationScale;
        boolean wroteSamples = false;

        try {
            while (readRecordOrEof(input, header, "V2 packet header")) {
                if ((header[0] & 0xFF) != GT3_SYNC_BYTE) {
                    throw new ReaderSupport.FormatException(
                            "Invalid GT3X packet synchronization byte");
                }
                int recordType = header[1] & 0xFF;
                long timestamp = unsignedLittleEndianInt(header, 2);
                int payloadSize = (header[6] & 0xFF) | ((header[7] & 0xFF) << 8);
                if (payloadSize > payload.length) {
                    payload = new byte[payloadSize];
                }
                readFully(input, payload, payloadSize, "V2 packet payload");
                int storedChecksum = input.read();
                if (storedChecksum == -1) {
                    throw new EOFException("Unexpected end of V2 packet checksum");
                }
                validateChecksum(header, payload, payloadSize, storedChecksum);

                if (recordType == PARAMETER_ID) {
                    accelerationScale = readParameterScale(
                            payload, payloadSize, accelerationScale);
                } else if ((recordType == ACTIVITY_ID || recordType == ACTIVITY2_ID)
                        && payloadSize > 1) {
                    if (!isValidAccelerationScale(accelerationScale)) {
                        throw new ReaderSupport.FormatException(
                                "No valid acceleration scale found before activity data");
                    }
                    int samplesWritten;
                    if (recordType == ACTIVITY_ID) {
                        samplesWritten = writePackedActivity(
                                payload, payloadSize, timestamp, metadata.sampleRate,
                                accelerationScale, packedSamples, writer);
                    } else {
                        samplesWritten = writeShortActivity(
                                payload, payloadSize, timestamp, metadata.sampleRate,
                                accelerationScale, writer);
                    }
                    wroteSamples |= samplesWritten > 0;
                }
            }
        } catch (EOFException error) {
            if (!wroteSamples) {
                throw error;
            }
            result.recordRecoverableError(
                    "Stopping at truncated GT3X data: " + error.getMessage());
        }
    }

    private static double readParameterScale(
            byte[] payload,
            int payloadSize,
            double currentScale)
            throws ReaderSupport.FormatException {
        if (payloadSize % 8 != 0) {
            throw new ReaderSupport.FormatException(
                    "GT3X parameter payload length is not a multiple of eight");
        }
        for (int offset = 0; offset < payloadSize; offset += 8) {
            if (payload[offset] == 0 && payload[offset + 2] == 55) {
                int encoded = (int) unsignedLittleEndianInt(payload, offset + 4);
                double decoded = decodeParameter(encoded);
                if (isValidAccelerationScale(decoded)) {
                    currentScale = decoded;
                }
            }
        }
        return currentScale;
    }

    private static int writePackedActivity(
            byte[] payload,
            int payloadSize,
            long timestamp,
            double sampleRate,
            double accelerationScale,
            double[] samples,
            NpyWriter writer) throws IOException, ReaderSupport.FormatException {
        int trailingBytes = payloadSize % 9;
        if (trailingBytes != 0 && trailingBytes != 5) {
            throw new ReaderSupport.FormatException(
                    "Packed GT3X activity payload has an incomplete sample");
        }

        long sampleIndex = 0;
        int pairedBytes = payloadSize - trailingBytes;
        for (int offset = 0; offset < pairedBytes; offset += 9) {
            decodePackedPair(payload, offset, accelerationScale, samples);
            for (int sample = 0; sample < 2; sample++) {
                int axisOffset = sample * 3;
                writeV2Sample(
                        writer,
                        timestamp,
                        sampleIndex++,
                        sampleRate,
                        roundToThousandth(samples[axisOffset]),
                        roundToThousandth(samples[axisOffset + 1]),
                        roundToThousandth(samples[axisOffset + 2]));
            }
        }
        if (trailingBytes == 5) {
            decodeFirstPackedSample(
                    payload, pairedBytes, accelerationScale, samples, 0);
            writeV2Sample(
                    writer,
                    timestamp,
                    sampleIndex++,
                    sampleRate,
                    roundToThousandth(samples[0]),
                    roundToThousandth(samples[1]),
                    roundToThousandth(samples[2]));
        }
        return (int) sampleIndex;
    }

    private static int writeShortActivity(
            byte[] payload,
            int payloadSize,
            long timestamp,
            double sampleRate,
            double accelerationScale,
            NpyWriter writer) throws IOException, ReaderSupport.FormatException {
        if (payloadSize % 6 != 0) {
            throw new ReaderSupport.FormatException(
                    "GT3X activity payload length is not a multiple of six");
        }

        long sampleIndex = 0;
        for (int offset = 0; offset < payloadSize; offset += 6) {
            float x = roundToThousandth(
                    littleEndianShort(payload, offset) / accelerationScale);
            float y = roundToThousandth(
                    littleEndianShort(payload, offset + 2) / accelerationScale);
            float z = roundToThousandth(
                    littleEndianShort(payload, offset + 4) / accelerationScale);
            writeV2Sample(writer, timestamp, sampleIndex++, sampleRate, x, y, z);
        }
        return (int) sampleIndex;
    }

    private static void writeV2Sample(
            NpyWriter writer,
            long timestamp,
            long sampleIndex,
            double sampleRate,
            float x,
            float y,
            float z) throws IOException {
        long timeMillis = timestamp * 1000
                + Math.round(1000d * sampleIndex / sampleRate);
        writer.write(TimeUnit.MILLISECONDS.toNanos(timeMillis), x, y, z);
    }

    private static void decodePackedPair(
            byte[] bytes,
            int offset,
            double accelerationScale,
            double[] samples) {
        decodeFirstPackedSample(bytes, offset, accelerationScale, samples, 0);
        decodeSecondPackedSample(bytes, offset, accelerationScale, samples, 3);
    }

    private static void decodeFirstPackedSample(
            byte[] bytes,
            int offset,
            double accelerationScale,
            double[] samples,
            int sampleOffset) {
        short y1 = signExtend12(
                ((bytes[offset] & 0xFF) << 4)
                | ((bytes[offset + 1] & 0xF0) >>> 4));
        short x1 = signExtend12(
                ((bytes[offset + 1] & 0x0F) << 8)
                | (bytes[offset + 2] & 0xFF));
        short z1 = signExtend12(
                ((bytes[offset + 3] & 0xFF) << 4)
                | ((bytes[offset + 4] & 0xF0) >>> 4));
        samples[sampleOffset] = x1 / accelerationScale;
        samples[sampleOffset + 1] = y1 / accelerationScale;
        samples[sampleOffset + 2] = z1 / accelerationScale;
    }

    private static void decodeSecondPackedSample(
            byte[] bytes,
            int offset,
            double accelerationScale,
            double[] samples,
            int sampleOffset) {
        short y2 = signExtend12(
                ((bytes[offset + 4] & 0x0F) << 8)
                | (bytes[offset + 5] & 0xFF));
        short x2 = signExtend12(
                ((bytes[offset + 6] & 0xFF) << 4)
                | ((bytes[offset + 7] & 0xF0) >>> 4));
        short z2 = signExtend12(
                ((bytes[offset + 7] & 0x0F) << 8)
                | (bytes[offset + 8] & 0xFF));
        samples[sampleOffset] = x2 / accelerationScale;
        samples[sampleOffset + 1] = y2 / accelerationScale;
        samples[sampleOffset + 2] = z2 / accelerationScale;
    }

    private static short signExtend12(int value) {
        return (short) ((value & 0x800) != 0 ? value | 0xF000 : value);
    }

    private static float roundToThousandth(double value) {
        return (float) (Math.round(value * 1000d) / 1000d);
    }

    private static boolean readRecordOrEof(
            InputStream input,
            byte[] target,
            String description) throws IOException, ReaderSupport.FormatException {
        int offset = 0;
        while (offset < target.length) {
            int count = input.read(target, offset, target.length - offset);
            if (count == -1) {
                if (offset == 0) {
                    return false;
                }
                throw new EOFException("Unexpected end of " + description);
            }
            offset += count;
        }
        return true;
    }

    private static void readFully(
            InputStream input,
            byte[] target,
            int length,
            String description) throws IOException, ReaderSupport.FormatException {
        int offset = 0;
        while (offset < length) {
            int count = input.read(target, offset, length - offset);
            if (count == -1) {
                throw new EOFException("Unexpected end of " + description);
            }
            offset += count;
        }
    }

    private static void validateChecksum(
            byte[] header,
            byte[] payload,
            int payloadSize,
            int storedChecksum) throws ReaderSupport.FormatException {
        int checksum = 0;
        for (byte value : header) {
            checksum ^= value & 0xFF;
        }
        for (int offset = 0; offset < payloadSize; offset++) {
            checksum ^= payload[offset] & 0xFF;
        }
        int expected = (~checksum) & 0xFF;
        if (expected != storedChecksum) {
            throw new ReaderSupport.FormatException("GT3X packet checksum mismatch");
        }
    }

    private static long unsignedLittleEndianInt(byte[] bytes, int offset) {
        return (long) (bytes[offset] & 0xFF)
                | ((long) (bytes[offset + 1] & 0xFF) << 8)
                | ((long) (bytes[offset + 2] & 0xFF) << 16)
                | ((long) (bytes[offset + 3] & 0xFF) << 24);
    }

    private static short littleEndianShort(byte[] bytes, int offset) {
        return (short) ((bytes[offset] & 0xFF) | (bytes[offset + 1] << 8));
    }

    private static double accelerationScaleForSerial(String serialNumber) {
        if (serialNumber.startsWith("NEO") || serialNumber.startsWith("CLE")) {
            return 341.0;
        }
        if (serialNumber.startsWith("MOS")) {
            return 256.0;
        }
        return -1;
    }

    private static boolean isValidAccelerationScale(double accelerationScale) {
        return Double.isFinite(accelerationScale)
                && accelerationScale >= MIN_ACCELERATION_SCALE
                && accelerationScale <= MAX_ACCELERATION_SCALE;
    }

    private static boolean isScaleConsistentWithRange(
            double accelerationScale,
            double accelerationMin,
            double accelerationMax) {
        if (Double.isNaN(accelerationMin) || Double.isNaN(accelerationMax)) {
            return true;
        }
        if (!Double.isFinite(accelerationMin)
                || !Double.isFinite(accelerationMax)
                || accelerationMin >= 0
                || accelerationMax <= 0) {
            return false;
        }
        double range = Math.max(Math.abs(accelerationMin), Math.abs(accelerationMax));
        double rawFullScale = accelerationScale * range;
        return rawFullScale >= MIN_RAW_FULL_SCALE
                && rawFullScale <= MAX_RAW_FULL_SCALE;
    }

    private static double decodeParameter(int value) {
        final double floatMaximum = 8388608.0;
        final int encodedMinimum = 0x00800000;
        final int encodedMaximum = 0x007FFFFF;
        final int significandMask = 0x00FFFFFF;

        if (value == encodedMaximum) {
            return Integer.MAX_VALUE;
        }
        if (value == encodedMinimum) {
            return -Integer.MAX_VALUE;
        }

        int exponent = value >>> 24;
        if ((exponent & 0x80) != 0) {
            exponent |= 0xFFFFFF00;
        }
        int significand = value & significandMask;
        if ((significand & encodedMinimum) != 0) {
            significand |= 0xFF000000;
        }
        return (significand / floatMaximum) * Math.pow(2.0, exponent);
    }

    private static long ticksToMilliseconds(long ticks) {
        return (ticks - 621355968000000000L) / 10000;
    }

    private static int getGT3XVersion(ZipFile zip) {
        if (zip.getEntry("info.txt") == null) {
            return INVALID_GT3_FILE;
        }
        if (zip.getEntry("activity.bin") != null
                && zip.getEntry("lux.bin") != null) {
            return VALID_GT3_V1_FILE;
        }
        if (zip.getEntry("log.bin") != null) {
            return VALID_GT3_V2_FILE;
        }
        return INVALID_GT3_FILE;
    }
}
