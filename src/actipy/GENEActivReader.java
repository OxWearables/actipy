
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

    // Specification of items to be written
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
            // LineReader rawAccReader = new LineReader(accFile);
            // Read header to determine mfrGain and mfrOffset values
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
                // header: "Recorded Data" (0), serialCode (1), seq num (2),
                // blockTime (3), unassigned (4), temp (5), batteryVolt (6),
                // deviceStatus (7), freq (8), data (9)
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

                // now process hex data
                data = readLine(rawAccReader);

                // raw reading values
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

                        // Update values to calibrated measure (taken from GENEActiv manual)
                        x = (xRaw * 100.0d - mfrOffset[0]) / mfrGain[0];
                        y = (yRaw * 100.0d - mfrOffset[1]) / mfrGain[1];
                        z = (zRaw * 100.0d - mfrOffset[2]) / mfrGain[2];

                        t = (double)blockTime + (double)i * (1.0 / freq) * 1000;  // Unix millis

                        writer.write(toItems(TimeUnit.MILLISECONDS.toNanos((long) t), x, y, z, temperature));

                        hexPosition += 12;
                        i++;

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
                        // if (blockCount == numBlocksTotal) {System.out.print("\n");}
                    }
                }

            }
            rawAccReader.close();

            statusOK = 1;

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

        // Write to info.txt file. Each line is a key:value pair.
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
     * Replicates bin file header, also calculates and returns
     * x/y/z gain/offset values along with number of pages of data in file bin
     * format described in GENEActiv manual ("Decoding .bin files", pg.27)
     * http://www.geneactiv.org/wp-content/uploads/2014/03/
     * geneactiv_instruction_manual_v1.2.pdf
     */
    private static int parseBinFileHeader(
            BufferedReader reader,
            int fileHeaderSize, int linesToAxesCalibration,
            double[] gainVals, int[] offsetVals) {
        // read first i lines in bin file to writer
        for (int i = 0; i < linesToAxesCalibration; i++) {
            readLine(reader);
        }
        // read axes calibration lines for gain and offset values
        // data like -> x gain:25548 \n x offset:574 ... Volts:300 \n Lux:800
        gainVals[0] = Double.parseDouble(readLine(reader).split(":")[1].trim()); // xGain
        offsetVals[0] = Integer.parseInt(readLine(reader).split(":")[1].trim()); // xOffset
        gainVals[1] = Double.parseDouble(readLine(reader).split(":")[1].trim()); // y
        offsetVals[1] = Integer.parseInt(readLine(reader).split(":")[1].trim()); // y
        gainVals[2] = Double.parseDouble(readLine(reader).split(":")[1].trim()); // z
        offsetVals[2] = Integer.parseInt(readLine(reader).split(":")[1].trim()); // z
        int volts = Integer.parseInt(readLine(reader).split(":")[1].trim()); // volts
        int lux = Integer.parseInt(readLine(reader).split(":")[1].trim()); // lux
        readLine(reader); // 9 blank
        readLine(reader); // 10 memory status header
        int numBlocksTotal = Integer.parseInt(readLine(reader).split(":")[1].trim()); // 11

        // ignore remaining header lines in bin file
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
        // input hex base is 16
        int rawVal = Integer.parseInt(data.substring(startPos, startPos + length), 16);
        int unsignedLimit = 4096; // 2^[length*4] #i.e. 3 hexBytes (12 bits)
                                    // limit = 4096
        int signedLimit = 2048; // 2^[length*(4-1)] #i.e. 3 hexBytes - 1 bit (11
                                // bits) limit = 2048
        if (rawVal > signedLimit) {
            rawVal = rawVal - unsignedLimit;
        }
        return rawVal;
    }


    // Convert LocalDateTime to epoch milliseconds (from 1970 epoch)
    private static long getEpochMillis(LocalDateTime date) {
        return date.toInstant(ZoneOffset.UTC).toEpochMilli();
    }


    private static long secs2Nanos(double num) {
        return (long) (TimeUnit.SECONDS.toNanos(1) * num);
    }


    private static Map<String, Object> toItems(long t, float x, float y, float z, float temperature) {
        Map<String, Object> items = new HashMap<String, Object>();
        items.put("time", t);
        items.put("x", x);
        items.put("y", y);
        items.put("z", z);
        items.put("temperature", temperature);
        return items;
    }


    private static Map<String, Object> toItems(long t, double x, double y, double z, double temperature) {
        return toItems(t, (float) x, (float) y, (float) z, (float) temperature);
    }


}
