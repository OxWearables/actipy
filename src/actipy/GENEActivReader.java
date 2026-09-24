
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.time.format.DateTimeFormatter;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import java.time.LocalDateTime;
import java.time.ZoneOffset;


public class GENEActivReader {

    // Keep field order aligned with NpyWriter's primitive row layouts.
    private static final Map<String, String> ITEM_NAMES_AND_TYPES;
    static{
        Map<String, String> itemNamesAndTypes = new LinkedHashMap<String, String>();
        itemNamesAndTypes.put("time", "Datetime");
        itemNamesAndTypes.put("x", "Float");
        itemNamesAndTypes.put("y", "Float");
        itemNamesAndTypes.put("z", "Float");
        itemNamesAndTypes.put("temperature", "Float");
        ITEM_NAMES_AND_TYPES = Collections.unmodifiableMap(itemNamesAndTypes);
    }

    public static void main(String[] args) {

        String accFile = null;
        String outDir = null;
        boolean verbose = false;

        // Parse args string. Example:
        // $ java GENEActivReader -i /path/to/inputFile.bin -o /path/to/outputDir -v
        for (int i = 0; i < args.length; i++) {
            if ("-i".equals(args[i]) && i < args.length - 1) {
                accFile = args[++i];
            } else if ("-o".equals(args[i]) && i < args.length - 1) {
                outDir = args[++i];
            } else if ("-v".equals(args[i])) {
                verbose = true;
            }
        }

        if (accFile == null) {
            System.out.println("ERROR: No input file specified.");
            System.exit(1);
        }
        if (outDir == null) {
            System.out.println("ERROR: No output directory specified.");
            System.exit(1);
        }

        int fileHeaderSize = 59;
        int linesToAxesCalibration = 47;
        int blockHeaderSize = 9;
        int statusOK = -1;
        double sampleRate = -1;
        int errCounter = 0;

        String outData = outDir + File.separator + "data.npy";
        NpyWriter writer = new NpyWriter(outData, ITEM_NAMES_AND_TYPES);

        try {
            BufferedReader rawAccReader = new BufferedReader(new FileReader(accFile));
            // Read the header to determine manufacturer gain and offset values.
            double[] mfrGain = new double[3];
            int[] mfrOffset = new int[3];
            int numBlocksTotal = parseBinFileHeader(rawAccReader, fileHeaderSize, linesToAxesCalibration, mfrGain, mfrOffset);

            int blockCount = 0;
            String header;
            long blockTime = 0;  // Unix millis
            double temperature = 0.0;
            double freq = 0.0;
            String data;
            String timeFmtStr = "yyyy-MM-dd HH:mm:ss:SSS";
            DateTimeFormatter timeFmt = DateTimeFormatter.ofPattern(timeFmtStr);

            while ((readLine(rawAccReader)) != null) {
                // Header lines: record marker, serial/sequence, timestamp, unused
                // metadata, temperature, battery/status, frequency, then payload.
                for (int i = 1; i < blockHeaderSize; i++) {
                    try {
                        header = readLine(rawAccReader);
                        if (i == 3) {
                            blockTime = LocalDateTime
                                        .parse(header.split("Time:")[1], timeFmt)
                                        .toInstant(ZoneOffset.UTC)
                                        .toEpochMilli();
                        } else if (i == 5) {
                            temperature = Double.parseDouble(header.split(":")[1]);
                        } else if (i == 8) {
                            freq = Double.parseDouble(header.split(":")[1]);
                        }
                    } catch (Exception e) {
                        errCounter++;
                        e.printStackTrace();
                        continue;
                    }
                }
                sampleRate = freq;

                data = readLine(rawAccReader);

                int hexPosition = 0;
                int xRaw = 0;
                int yRaw = 0;
                int zRaw = 0;
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;
                double t = 0.0;

                int i = 0;
                while (hexPosition < data.length()) {

                    try {

                        xRaw = getSignedIntFromHex(data, hexPosition, 3);
                        yRaw = getSignedIntFromHex(data, hexPosition + 3, 3);
                        zRaw = getSignedIntFromHex(data, hexPosition + 6, 3);
                        // todo *** read in light[36:46] (10 bits to signed int) and
                        // button[47] (bool) values...

                        // Apply the gain and offset values from the GENEActiv header.
                        x = (xRaw * 100.0d - mfrOffset[0]) / mfrGain[0];
                        y = (yRaw * 100.0d - mfrOffset[1]) / mfrGain[1];
                        z = (zRaw * 100.0d - mfrOffset[2]) / mfrGain[2];

                        t = (double)blockTime + (double)i * (1.0 / freq) * 1000;  // Unix milliseconds.

                        writer.write(
                                TimeUnit.MILLISECONDS.toNanos((long) t),
                                (float) x, (float) y, (float) z, (float) temperature);

                        hexPosition += 12;
                        i++;

                    } catch (NpyWriter.SchemaMismatchException e) {
                        throw e;
                    } catch (Exception e) {
                        errCounter++;
                        e.printStackTrace();
                        break;  // rest of this block could be corrupted
                    }

                }

                // Progress bar
                blockCount++;
                if (verbose) {
                    if ((blockCount % 10000 == 0) || (blockCount == numBlocksTotal)) {
                        System.out.print("Reading file... " + (blockCount * 100 / numBlocksTotal) + "%\r");
                    }
                }

            }
            rawAccReader.close();

            statusOK = 1;

        } catch (NpyWriter.SchemaMismatchException e) {
            throw e;
        } catch (Exception e) {
            e.printStackTrace();
            statusOK = 0;

        } finally {
            try{
                writer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        Map<String, String> info = new HashMap<String, String>();
        info.put("ReadOK", String.valueOf(statusOK));
        info.put("ReadErrors", String.valueOf(errCounter));
        info.put("SampleRate", String.valueOf(sampleRate));

        // Persist reader status metadata alongside the converted data.
        String outInfo = outDir + File.separator + "info.txt";
        try {
            FileWriter file = new FileWriter(outInfo);
            for (Map.Entry<String, String> entry : info.entrySet()) {
                file.write(entry.getKey() + ":" + entry.getValue() + "\n");
            }
            file.flush();
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return;

    }


    /**
     * Reads the .bin header, returning x/y/z gain and offset values together
     * with the number of data blocks, following the GENEActiv manual
     * ("Decoding .bin files", p. 27).
     * http://www.geneactiv.org/wp-content/uploads/2014/03/
     * geneactiv_instruction_manual_v1.2.pdf
     */
    private static int parseBinFileHeader(
            BufferedReader reader,
            int fileHeaderSize, int linesToAxesCalibration,
            double[] gainVals, int[] offsetVals) {
        for (int i = 0; i < linesToAxesCalibration; i++) {
            readLine(reader);
        }
        // The next lines contain alternating gain and offset values for x, y, and z.
        gainVals[0] = Double.parseDouble(readLine(reader).split(":")[1].trim());
        offsetVals[0] = Integer.parseInt(readLine(reader).split(":")[1].trim());
        gainVals[1] = Double.parseDouble(readLine(reader).split(":")[1].trim());
        offsetVals[1] = Integer.parseInt(readLine(reader).split(":")[1].trim());
        gainVals[2] = Double.parseDouble(readLine(reader).split(":")[1].trim());
        offsetVals[2] = Integer.parseInt(readLine(reader).split(":")[1].trim());
        int volts = Integer.parseInt(readLine(reader).split(":")[1].trim()); // voltage
        int lux = Integer.parseInt(readLine(reader).split(":")[1].trim()); // illuminance
        readLine(reader); // blank line
        readLine(reader); // memory status header
        int numBlocksTotal = Integer.parseInt(readLine(reader).split(":")[1].trim());

        // Skip the rest of the fixed-size header.
        for (int i = 0; i < fileHeaderSize - linesToAxesCalibration - 11; i++) {
            readLine(reader);
        }
        return numBlocksTotal;

    }


    private static String readLine(BufferedReader reader) {
        String line = "";
        try {
            line = reader.readLine();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return line;
    }


    private static int getSignedIntFromHex(String data, int startPos, int length) {
        int rawVal = Integer.parseInt(data.substring(startPos, startPos + length), 16);
        int unsignedLimit = 4096; // 2^[length*4] #i.e. 3 hexBytes (12 bits)
        int signedLimit = 2048; // 2^[length*(4-1)] #i.e. 3 hexBytes - 1 bit (11
                                // bits) limit = 2048
        if (rawVal >= signedLimit) {
            rawVal = rawVal - unsignedLimit;
        }
        return rawVal;
    }


    private static long getEpochMillis(LocalDateTime date) {
        return date.toInstant(ZoneOffset.UTC).toEpochMilli();
    }


    private static long secs2Nanos(double num) {
        return (long) (TimeUnit.SECONDS.toNanos(1) * num);
    }
}
