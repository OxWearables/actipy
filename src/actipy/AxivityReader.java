import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.time.DateTimeException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.concurrent.TimeUnit;


public class AxivityReader {

    private static final int BLOCK_SIZE = 512;
    private static final int SAMPLE_PAYLOAD_SIZE = 480;

    private enum Format {
        AX3(NpyWriter.Layout.XYZTL),
        AX6(NpyWriter.Layout.XYZ_GYRO_TL);

        final NpyWriter.Layout outputLayout;

        Format(NpyWriter.Layout outputLayout) {
            this.outputLayout = outputLayout;
        }
    }

    private static final class DataBlock {
        final ByteBuffer bytes;
        final Format format;
        final int numAxes;
        final int packing;
        final int bytesPerSample;
        final int sampleCount;
        final boolean sampleCountClamped;
        final float sampleRate;
        final double startTime;
        final float temperature;
        final float light;
        final int accelerationUnit;
        final float gyroscopeUnit;

        private DataBlock(
                ByteBuffer bytes,
                Format format,
                int numAxes,
                int packing,
                int bytesPerSample,
                int sampleCount,
                boolean sampleCountClamped,
                float sampleRate,
                double startTime,
                float temperature,
                float light,
                int accelerationUnit,
                float gyroscopeUnit) {
            this.bytes = bytes;
            this.format = format;
            this.numAxes = numAxes;
            this.packing = packing;
            this.bytesPerSample = bytesPerSample;
            this.sampleCount = sampleCount;
            this.sampleCountClamped = sampleCountClamped;
            this.sampleRate = sampleRate;
            this.startTime = startTime;
            this.temperature = temperature;
            this.light = light;
            this.accelerationUnit = accelerationUnit;
            this.gyroscopeUnit = gyroscopeUnit;
        }

        static DataBlock parse(byte[] raw) throws ReaderSupport.FormatException {
            ByteBuffer block = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            byte marker0 = block.get(0);
            byte marker1 = block.get(1);
            if (marker0 == 'M' && marker1 == 'D') {
                return null;
            }
            if (marker0 != 'A' || marker1 != 'X') {
                return null;
            }

            int rateCode = block.get(24) & 0xFF;
            if (rateCode != 0) {
                int checksum = 0;
                for (int index = 0; index < BLOCK_SIZE / 2; index++) {
                    checksum = (short) (checksum + block.getShort(index * 2));
                }
                if (checksum != 0) {
                    throw new ReaderSupport.FormatException("CWA block checksum mismatch");
                }
            }

            int axesAndPacking = block.get(25) & 0xFF;
            int numAxes = (axesAndPacking >>> 4) & 0x0F;
            int packing = axesAndPacking & 0x0F;
            Format format;
            if (numAxes >= 6) {
                format = Format.AX6;
            } else if (numAxes >= 3) {
                format = Format.AX3;
            } else {
                throw new ReaderSupport.FormatException(
                        "CWA block has fewer than three axes");
            }

            int bytesPerSample;
            if (packing == 2) {
                bytesPerSample = 2 * numAxes;
            } else if (packing == 0 && numAxes == 3) {
                bytesPerSample = 4;
            } else {
                throw new ReaderSupport.FormatException(
                        "Unsupported CWA packing and axis combination");
            }

            int declaredSampleCount = unsignedShort(block, 28);
            int maxSamples = SAMPLE_PAYLOAD_SIZE / bytesPerSample;
            int sampleCount = Math.min(declaredSampleCount, maxSamples);

            float sampleRate;
            float offsetStart;
            if (rateCode == 0) {
                sampleRate = block.getShort(26);
                offsetStart = 0;
            } else {
                short timestampOffset = block.getShort(26);
                sampleRate = 3200.0f / (1 << (15 - (rateCode & 15)));
                offsetStart = -timestampOffset / sampleRate;
            }
            if (!Float.isFinite(sampleRate) || sampleRate <= 0) {
                throw new ReaderSupport.FormatException(
                        "CWA block has no positive finite sample rate");
            }

            int timeInfo = (int) unsignedInt(block, 14);
            long blockTime;
            try {
                blockTime = cwaTimestamp(timeInfo);
            } catch (DateTimeException error) {
                throw new ReaderSupport.FormatException(
                        "CWA block has an invalid timestamp", error);
            }
            blockTime += (long) Math.floor(offsetStart);
            offsetStart -= (float) Math.floor(offsetStart);

            int rawLight = unsignedShort(block, 18);
            float light = (float) Math.pow(10, (rawLight & 0x3FF) / 341.0);
            float temperature = (float) (((unsignedShort(block, 20) & 0x3FF)
                    * 150.0 - 20500) / 1000);
            int accelerationUnit = 1 << (8 + ((rawLight >>> 13) & 0x07));
            int gyroscopeRange = 2000;
            if (((rawLight >>> 10) & 0x07) != 0) {
                gyroscopeRange = 8000 / (1 << ((rawLight >>> 10) & 0x07));
            }
            float gyroscopeUnit = 32768.0f / gyroscopeRange;

            return new DataBlock(
                    block,
                    format,
                    numAxes,
                    packing,
                    bytesPerSample,
                    sampleCount,
                    declaredSampleCount > maxSamples,
                    sampleRate,
                    (double) blockTime + offsetStart,
                    temperature,
                    light,
                    accelerationUnit,
                    gyroscopeUnit);
        }
    }

    private static final class BlockDecoder {
        private final NpyWriter writer;
        private final Format expectedFormat;
        private final ReaderSupport.Result result;
        private double lastBlockTime;

        BlockDecoder(
                NpyWriter writer,
                Format expectedFormat,
                ReaderSupport.Result result) {
            this.writer = writer;
            this.expectedFormat = expectedFormat;
            this.result = result;
        }

        int write(DataBlock block) throws IOException, ReaderSupport.FormatException {
            if (block.format != expectedFormat) {
                throw new ReaderSupport.FormatException(
                        "CWA axis layout changes from "
                        + expectedFormat + " to " + block.format);
            }
            if (block.sampleCountClamped) {
                result.readErrors++;
                System.err.println("Capping malformed CWA sample count at payload capacity");
            }

            double blockStartTime = block.startTime;
            double blockEndTime = blockStartTime
                    + (float) block.sampleCount / block.sampleRate;
            if (lastBlockTime != 0 && blockStartTime - lastBlockTime < 1.0) {
                blockStartTime = lastBlockTime;
            }
            lastBlockTime = blockEndTime;
            result.sampleRate = block.sampleRate;

            short[] values = new short[block.numAxes];
            int accelerationAxis = block.format == Format.AX6 ? 3 : 0;
            for (int sampleIndex = 0; sampleIndex < block.sampleCount; sampleIndex++) {
                decodeSample(block, sampleIndex, values);
                float ax = values[accelerationAxis] / (float) block.accelerationUnit;
                float ay = values[accelerationAxis + 1] / (float) block.accelerationUnit;
                float az = values[accelerationAxis + 2] / (float) block.accelerationUnit;
                double time = blockStartTime
                        + sampleIndex * (blockEndTime - blockStartTime)
                        / block.sampleCount;
                long timeNanos = TimeUnit.MILLISECONDS.toNanos((long) (time * 1000));

                if (block.format == Format.AX6) {
                    float gx = values[0] / block.gyroscopeUnit;
                    float gy = values[1] / block.gyroscopeUnit;
                    float gz = values[2] / block.gyroscopeUnit;
                    writer.write(
                            timeNanos,
                            ax, ay, az,
                            gx, gy, gz,
                            block.temperature,
                            block.light);
                } else {
                    writer.write(
                            timeNanos,
                            ax, ay, az,
                            block.temperature,
                            block.light);
                }
            }
            return block.sampleCount;
        }

        private void decodeSample(DataBlock block, int sampleIndex, short[] values) {
            if (block.packing == 0) {
                long packed = unsignedInt(block.bytes, 30 + 4 * sampleIndex);
                int exponent = (int) ((packed >>> 30) & 0x03);
                values[0] = (short) ((short) (0xFFFFFFC0 & (packed << 6))
                        >> (6 - exponent));
                values[1] = (short) ((short) (0xFFFFFFC0 & (packed >> 4))
                        >> (6 - exponent));
                values[2] = (short) ((short) (0xFFFFFFC0 & (packed >> 14))
                        >> (6 - exponent));
            } else {
                int sampleOffset = 30 + block.bytesPerSample * sampleIndex;
                for (int axis = 0; axis < block.numAxes; axis++) {
                    values[axis] = block.bytes.getShort(sampleOffset + 2 * axis);
                }
            }
        }
    }

    public static void main(String[] args) {
        ReaderSupport.run(args, AxivityReader::convert);
    }

    private static void convert(
            ReaderSupport.Options options,
            ReaderSupport.Result result) throws Exception {
        try (FileInputStream input = new FileInputStream(options.inputFile);
             FileChannel channel = input.getChannel()) {
            long totalBlocks = channel.size() / BLOCK_SIZE;
            long blocksRead = 0;
            byte[] raw = new byte[BLOCK_SIZE];
            DataBlock firstDataBlock = null;

            while (firstDataBlock == null && readBlockOrEof(channel, raw)) {
                blocksRead++;
                firstDataBlock = parseDataBlock(raw, result);
            }
            if (firstDataBlock == null) {
                throw new ReaderSupport.FormatException("No valid CWA data block found");
            }

            try (NpyWriter writer = new NpyWriter(
                    options.dataPath(), firstDataBlock.format.outputLayout)) {
                BlockDecoder decoder = new BlockDecoder(
                        writer, firstDataBlock.format, result);
                int samplesWritten = decoder.write(firstDataBlock);

                try {
                    while (readBlockOrEof(channel, raw)) {
                        blocksRead++;
                        DataBlock block = parseDataBlock(raw, result);
                        if (block != null) {
                            samplesWritten += decoder.write(block);
                        }

                        if (options.verbose
                                && (blocksRead % 10000 == 0
                                || blocksRead == totalBlocks)) {
                            int percent = totalBlocks > 0
                                    ? (int) (blocksRead * 100 / totalBlocks)
                                    : 100;
                            System.out.print("Reading file... " + percent + "%\r");
                        }
                    }
                } catch (EOFException error) {
                    if (samplesWritten == 0) {
                        throw error;
                    }
                    result.recordRecoverableError(
                            "Stopping at truncated CWA data: " + error.getMessage());
                }
            }
        }
    }

    private static boolean readBlockOrEof(FileChannel channel, byte[] target)
            throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(target);
        while (buffer.hasRemaining()) {
            int count = channel.read(buffer);
            if (count == -1) {
                if (buffer.position() == 0) {
                    return false;
                }
                throw new EOFException("Unexpected partial CWA block");
            }
        }
        return true;
    }

    private static DataBlock parseDataBlock(
            byte[] raw,
            ReaderSupport.Result result) {
        try {
            return DataBlock.parse(raw);
        } catch (ReaderSupport.FormatException error) {
            result.readErrors++;
            System.err.println("Skipping malformed CWA block: " + error.getMessage());
            return null;
        }
    }

    private static LocalDateTime cwaLocalDateTime(int value) {
        int year = ((value >>> 26) & 0x3F) + 2000;
        int month = (value >>> 22) & 0x0F;
        int day = (value >>> 17) & 0x1F;
        int hour = (value >>> 12) & 0x1F;
        int minute = (value >>> 6) & 0x3F;
        int second = value & 0x3F;
        return LocalDateTime.of(year, month, day, hour, minute, second);
    }

    private static long cwaTimestamp(int value) {
        return cwaLocalDateTime(value).toEpochSecond(ZoneOffset.UTC);
    }

    private static long unsignedInt(ByteBuffer buffer, int position) {
        return buffer.getInt(position) & 0xFFFFFFFFL;
    }

    private static int unsignedShort(ByteBuffer buffer, int position) {
        return buffer.getShort(position) & 0xFFFF;
    }
}
