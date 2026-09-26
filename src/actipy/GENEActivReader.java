import java.io.BufferedReader;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.concurrent.TimeUnit;


public class GENEActivReader {

    private static final int FILE_HEADER_LINES = 59;
    private static final int LINES_TO_CALIBRATION = 47;
    private static final int PAGE_LINES = 10;
    private static final int SAMPLES_PER_PAGE = 300;
    private static final int HEX_CHARACTERS_PER_SAMPLE = 12;
    private static final int PAGE_PAYLOAD_LENGTH =
            SAMPLES_PER_PAGE * HEX_CHARACTERS_PER_SAMPLE;
    private static final DateTimeFormatter PAGE_TIME_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss:SSS");

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
                     new FileInputStream(options.inputFile), StandardCharsets.US_ASCII))) {
            Calibration calibration = readCalibration(reader);
            validateCalibration(calibration);

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
                            page, calibration, writer, result);
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
            long startTimeMillis = LocalDateTime.parse(timestamp, PAGE_TIME_FORMAT)
                    .toInstant(ZoneOffset.UTC)
                    .toEpochMilli();
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
        } catch (NumberFormatException | DateTimeParseException error) {
            throw new ReaderSupport.FormatException(
                    "Invalid page timestamp or numeric metadata", error);
        }
    }

    private static int writePageSamples(
            PageHeader page,
            Calibration calibration,
            NpyWriter writer,
            ReaderSupport.Result result) throws IOException {
        if (page.payload.length() != PAGE_PAYLOAD_LENGTH
                || !isHexadecimal(page.payload)) {
            result.recordRecoverableError(
                    "Skipping invalid GENEActiv page payload: expected "
                    + PAGE_PAYLOAD_LENGTH + " hexadecimal characters");
            return 0;
        }

        for (int sampleIndex = 0; sampleIndex < SAMPLES_PER_PAGE; sampleIndex++) {
            int position = sampleIndex * HEX_CHARACTERS_PER_SAMPLE;
            int xRaw = signed12Bit(page.payload, position);
            int yRaw = signed12Bit(page.payload, position + 3);
            int zRaw = signed12Bit(page.payload, position + 6);
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

    private static boolean isHexadecimal(String value) {
        for (int index = 0; index < value.length(); index++) {
            if (Character.digit(value.charAt(index), 16) < 0) {
                return false;
            }
        }
        return true;
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

    private static int signed12Bit(String data, int position) {
        int rawValue = Integer.parseInt(data.substring(position, position + 3), 16);
        return rawValue >= 2048 ? rawValue - 4096 : rawValue;
    }
}
