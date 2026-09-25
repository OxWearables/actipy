import java.io.BufferedReader;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.DateTimeException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.concurrent.TimeUnit;


public class GENEActivReader {

    private static final int FILE_HEADER_LINES = 59;
    private static final int LINES_TO_CALIBRATION = 47;
    private static final int PAGE_LINES = 10;
    private static final int SAMPLES_PER_PAGE = 300;
    private static final int HEX_CHARACTERS_PER_SAMPLE = 12;
    private static final int PAGE_PAYLOAD_LENGTH =
            SAMPLES_PER_PAGE * HEX_CHARACTERS_PER_SAMPLE;
    private static final int INPUT_BUFFER_SIZE = 64 * 1024;

    private static final class Calibration {
        final double[] gains = new double[3];
        final int[] offsets = new int[3];
        int expectedPages;
    }

    private static final class PageHeader {
        final long startTimeMillis;
        final double temperature;
        final double sampleRate;
        final String payload;

        PageHeader(
                long startTimeMillis,
                double temperature,
                double sampleRate,
                String payload) {
            this.startTimeMillis = startTimeMillis;
            this.temperature = temperature;
            this.sampleRate = sampleRate;
            this.payload = payload;
        }
    }

    public static void main(String[] args) {
        ReaderSupport.run(args, GENEActivReader::convert);
    }

    private static void convert(
            ReaderSupport.Options options,
            ReaderSupport.Result result) throws Exception {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                     new FileInputStream(options.inputFile), StandardCharsets.US_ASCII),
                     INPUT_BUFFER_SIZE)) {
            Calibration calibration = readCalibration(reader);
            validateCalibration(calibration);
            short[] decodedSamples = new short[SAMPLES_PER_PAGE * 3];

            try (NpyWriter writer = new NpyWriter(
                    options.dataPath(), NpyWriter.Layout.XYZT)) {
                int pageCount = 0;
                int validPageCount = 0;
                int samplesWritten = 0;
                boolean truncatedPage = false;
                while (true) {
                    String[] pageLines;
                    try {
                        pageLines = readPage(reader);
                    } catch (EOFException error) {
                        if (samplesWritten == 0) {
                            throw error;
                        }
                        result.recordRecoverableError(
                                "Stopping at truncated GENEActiv data: "
                                + error.getMessage());
                        truncatedPage = true;
                        break;
                    }
                    if (pageLines == null) {
                        break;
                    }
                    pageCount++;
                    PageHeader page;
                    try {
                        page = parsePageHeader(pageLines);
                    } catch (ReaderSupport.FormatException error) {
                        result.readErrors++;
                        System.err.println("Skipping malformed GENEActiv page "
                                + pageCount + ": " + error.getMessage());
                        continue;
                    }

                    int pageSamples = writePageSamples(
                            page, calibration, decodedSamples, writer, result);
                    if (pageSamples > 0) {
                        validPageCount++;
                        samplesWritten += pageSamples;
                        result.sampleRate = page.sampleRate;
                    }

                    if (options.verbose
                            && (pageCount % 10000 == 0
                            || pageCount == calibration.expectedPages)) {
                        int percent = calibration.expectedPages > 0
                                ? pageCount * 100 / calibration.expectedPages
                                : 100;
                        System.out.print("Reading file... " + percent + "%\r");
                    }
                }

                if ((pageCount > 0 || calibration.expectedPages > 0)
                        && validPageCount == 0) {
                    throw new ReaderSupport.FormatException(
                            "No valid GENEActiv pages were decoded");
                }
                if (pageCount != calibration.expectedPages && !truncatedPage) {
                    result.recordRecoverableError(
                            "GENEActiv page count differs from header: expected "
                            + calibration.expectedPages + " but found " + pageCount);
                }
            }
        }
    }

    private static Calibration readCalibration(BufferedReader reader)
            throws IOException, ReaderSupport.FormatException {
        Calibration calibration = new Calibration();
        for (int line = 0; line < LINES_TO_CALIBRATION; line++) {
            requireLine(reader, "GENEActiv file header");
        }

        calibration.gains[0] = parseDoubleValue(
                requireLine(reader, "x gain"), "x gain");
        calibration.offsets[0] = parseIntValue(
                requireLine(reader, "x offset"), "x offset");
        calibration.gains[1] = parseDoubleValue(
                requireLine(reader, "y gain"), "y gain");
        calibration.offsets[1] = parseIntValue(
                requireLine(reader, "y offset"), "y offset");
        calibration.gains[2] = parseDoubleValue(
                requireLine(reader, "z gain"), "z gain");
        calibration.offsets[2] = parseIntValue(
                requireLine(reader, "z offset"), "z offset");

        parseIntValue(requireLine(reader, "voltage"), "Volts");
        parseIntValue(requireLine(reader, "illuminance"), "Lux");
        requireLine(reader, "header separator");
        requireLine(reader, "memory status header");
        calibration.expectedPages = parseIntValue(
                requireLine(reader, "page count"), "Number of Pages");

        int consumed = LINES_TO_CALIBRATION + 11;
        for (int line = consumed; line < FILE_HEADER_LINES; line++) {
            requireLine(reader, "GENEActiv file header");
        }
        return calibration;
    }

    private static void validateCalibration(Calibration calibration)
            throws ReaderSupport.FormatException {
        for (int axis = 0; axis < calibration.gains.length; axis++) {
            if (!Double.isFinite(calibration.gains[axis])
                    || calibration.gains[axis] == 0) {
                throw new ReaderSupport.FormatException(
                        "Calibration gain must be finite and non-zero for axis " + axis);
            }
        }
        if (calibration.expectedPages < 0) {
            throw new ReaderSupport.FormatException("Page count must not be negative");
        }
    }

    private static String[] readPage(BufferedReader reader) throws IOException {
        String firstLine = reader.readLine();
        if (firstLine == null) {
            return null;
        }

        String[] lines = new String[PAGE_LINES];
        lines[0] = firstLine;
        for (int line = 1; line < PAGE_LINES; line++) {
            lines[line] = reader.readLine();
            if (lines[line] == null) {
                throw new EOFException("Unexpected end of GENEActiv page");
            }
        }
        return lines;
    }

    private static PageHeader parsePageHeader(String[] lines)
            throws ReaderSupport.FormatException {
        if (!"Recorded Data".equals(lines[0])) {
            throw new ReaderSupport.FormatException("Missing Recorded Data marker");
        }
        try {
            String timestamp = valueAfterColon(lines[3], "Page Time");
            long startTimeMillis = parsePageTime(timestamp);
            double temperature = Double.parseDouble(
                    valueAfterColon(lines[5], "Temperature"));
            double sampleRate = Double.parseDouble(
                    valueAfterColon(lines[8], "Measurement Frequency"));
            if (!Double.isFinite(temperature)) {
                throw new ReaderSupport.FormatException("Temperature must be finite");
            }
            if (!Double.isFinite(sampleRate) || sampleRate <= 0) {
                throw new ReaderSupport.FormatException(
                        "Measurement Frequency must be positive and finite");
            }
            return new PageHeader(
                    startTimeMillis, temperature, sampleRate, lines[9]);
        } catch (NumberFormatException | DateTimeException error) {
            throw new ReaderSupport.FormatException(
                    "Invalid page timestamp or numeric metadata", error);
        }
    }

    private static int writePageSamples(
            PageHeader page,
            Calibration calibration,
            short[] decodedSamples,
            NpyWriter writer,
            ReaderSupport.Result result) throws IOException {
        if (page.payload.length() != PAGE_PAYLOAD_LENGTH
                || !decodePageSamples(page.payload, decodedSamples)) {
            result.recordRecoverableError(
                    "Skipping invalid GENEActiv page payload: expected "
                    + PAGE_PAYLOAD_LENGTH + " hexadecimal characters");
            return 0;
        }

        for (int sampleIndex = 0; sampleIndex < SAMPLES_PER_PAGE; sampleIndex++) {
            int position = sampleIndex * 3;
            int xRaw = decodedSamples[position];
            int yRaw = decodedSamples[position + 1];
            int zRaw = decodedSamples[position + 2];
            double x = (xRaw * 100.0d - calibration.offsets[0])
                    / calibration.gains[0];
            double y = (yRaw * 100.0d - calibration.offsets[1])
                    / calibration.gains[1];
            double z = (zRaw * 100.0d - calibration.offsets[2])
                    / calibration.gains[2];
            long timeMillis = (long) (page.startTimeMillis
                    + sampleIndex * 1000d / page.sampleRate);
            writer.write(
                    TimeUnit.MILLISECONDS.toNanos(timeMillis),
                    (float) x,
                    (float) y,
                    (float) z,
                    (float) page.temperature);
        }
        return SAMPLES_PER_PAGE;
    }

    private static boolean decodePageSamples(String payload, short[] decodedSamples) {
        int decodedIndex = 0;
        for (int sample = 0; sample < SAMPLES_PER_PAGE; sample++) {
            int sampleStart = sample * HEX_CHARACTERS_PER_SAMPLE;
            for (int component = 0; component < 4; component++) {
                int position = sampleStart + component * 3;
                int high = hexDigit(payload.charAt(position));
                int middle = hexDigit(payload.charAt(position + 1));
                int low = hexDigit(payload.charAt(position + 2));
                if ((high | middle | low) < 0) {
                    return false;
                }
                if (component < 3) {
                    int rawValue = (high << 8) | (middle << 4) | low;
                    decodedSamples[decodedIndex++] = (short) (
                            rawValue >= 2048 ? rawValue - 4096 : rawValue);
                }
            }
        }
        return true;
    }

    private static int hexDigit(char value) {
        if (value >= '0' && value <= '9') {
            return value - '0';
        }
        if (value >= 'A' && value <= 'F') {
            return value - 'A' + 10;
        }
        if (value >= 'a' && value <= 'f') {
            return value - 'a' + 10;
        }
        return -1;
    }

    private static long parsePageTime(String timestamp) {
        if (timestamp.length() != 23
                || timestamp.charAt(4) != '-'
                || timestamp.charAt(7) != '-'
                || timestamp.charAt(10) != ' '
                || timestamp.charAt(13) != ':'
                || timestamp.charAt(16) != ':'
                || timestamp.charAt(19) != ':') {
            throw new DateTimeException("Invalid GENEActiv page timestamp");
        }

        int year = fixedDecimal(timestamp, 0, 4);
        int month = fixedDecimal(timestamp, 5, 2);
        int day = fixedDecimal(timestamp, 8, 2);
        int hour = fixedDecimal(timestamp, 11, 2);
        int minute = fixedDecimal(timestamp, 14, 2);
        int second = fixedDecimal(timestamp, 17, 2);
        int millis = fixedDecimal(timestamp, 20, 3);
        return LocalDateTime.of(
                year, month, day, hour, minute, second, millis * 1_000_000)
                .toEpochSecond(ZoneOffset.UTC) * 1000 + millis;
    }

    private static int fixedDecimal(String value, int offset, int length) {
        int parsed = 0;
        for (int index = offset; index < offset + length; index++) {
            char digit = value.charAt(index);
            if (digit < '0' || digit > '9') {
                throw new DateTimeException("Invalid GENEActiv page timestamp");
            }
            parsed = parsed * 10 + digit - '0';
        }
        return parsed;
    }

    private static String requireLine(BufferedReader reader, String description)
            throws IOException, ReaderSupport.FormatException {
        String line = reader.readLine();
        if (line == null) {
            throw new ReaderSupport.FormatException(
                    "Unexpected end of " + description);
        }
        return line;
    }

    private static String valueAfterColon(String line, String expectedKey)
            throws ReaderSupport.FormatException {
        int separator = line.indexOf(':');
        if (separator < 0 || !expectedKey.equals(line.substring(0, separator).trim())) {
            throw new ReaderSupport.FormatException(
                    "Expected " + expectedKey + " header field");
        }
        return line.substring(separator + 1).trim();
    }

    private static double parseDoubleValue(String line, String expectedKey)
            throws ReaderSupport.FormatException {
        try {
            return Double.parseDouble(valueAfterColon(line, expectedKey));
        } catch (NumberFormatException error) {
            throw new ReaderSupport.FormatException(
                    "Invalid numeric value for " + expectedKey, error);
        }
    }

    private static int parseIntValue(String line, String expectedKey)
            throws ReaderSupport.FormatException {
        try {
            return Integer.parseInt(valueAfterColon(line, expectedKey));
        } catch (NumberFormatException error) {
            throw new ReaderSupport.FormatException(
                    "Invalid numeric value for " + expectedKey, error);
        }
    }

}
