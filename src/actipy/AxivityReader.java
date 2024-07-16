/**
 * Based on the OpenMovement implementation:
 * https://github.com/digitalinteraction/openmovement/blob/72c992b0ea524275d898e86181c5b38a9622c529/Software/AX3/cwa-convert/java/src/newcastle/cwa/CwaBlock.java
*/
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.nio.ByteOrder;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.concurrent.TimeUnit;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.io.IOException;


public class AxivityReader {

    private static final int BLOCKSIZE = 512;

    // Specification of items to be written
    private static final Map<String, String> ITEM_NAMES_AND_TYPES_AX3;
    private static final Map<String, String> ITEM_NAMES_AND_TYPES_AX6;
    static{
        Map<String, String> itemNamesAndTypes = new LinkedHashMap<String, String>();
        itemNamesAndTypes.put("time", "Datetime");
        itemNamesAndTypes.put("x", "Float");
        itemNamesAndTypes.put("y", "Float");
        itemNamesAndTypes.put("z", "Float");
        itemNamesAndTypes.put("temperature", "Float");
        itemNamesAndTypes.put("light", "Float");
        ITEM_NAMES_AND_TYPES_AX3 = Collections.unmodifiableMap(itemNamesAndTypes);
    }
    static{
        Map<String, String> itemNamesAndTypes = new LinkedHashMap<String, String>();
        itemNamesAndTypes.put("time", "Datetime");
        itemNamesAndTypes.put("x", "Float");
        itemNamesAndTypes.put("y", "Float");
        itemNamesAndTypes.put("z", "Float");
        itemNamesAndTypes.put("gyro_x", "Float");
        itemNamesAndTypes.put("gyro_y", "Float");
        itemNamesAndTypes.put("gyro_z", "Float");
        itemNamesAndTypes.put("temperature", "Float");
        itemNamesAndTypes.put("light", "Float");
        ITEM_NAMES_AND_TYPES_AX6 = Collections.unmodifiableMap(itemNamesAndTypes);
    }

    public static void main(String[] args) {

        int statusOK = -1;
        String accFile = null;
        String outDir = null;
        boolean verbose = false;

        // Parse args string. Example:
        // $ java AxivityReader -i /path/to/inputFile.bin -o /path/to/outputDir -v
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

        boolean hasGyro = detectGyro(accFile);
        Map<String, String> item_names_and_types = hasGyro ? ITEM_NAMES_AND_TYPES_AX6 : ITEM_NAMES_AND_TYPES_AX3;

        String outData = outDir + File.separator + "data.npy";
        NpyWriter writer = new NpyWriter(outData, item_names_and_types);

        BlockParser blockParser = new BlockParser(writer);

        try(FileInputStream accStream = new FileInputStream(accFile);
            FileChannel accChannel = accStream.getChannel();) {

            int blockCount = 0;
            long numBlocksTotal = accChannel.size() / BLOCKSIZE;
            ByteBuffer block = ByteBuffer.allocate(BLOCKSIZE);

            while (accChannel.read(block) != -1) {

                blockParser.parse(block);
                block.clear();

                // Progress bar
                blockCount++;
                if (verbose) {
                    if ((blockCount % 10000 == 0) || (blockCount == numBlocksTotal)) {
                        System.out.print("Reading file... " + (blockCount * 100 / numBlocksTotal) + "%\r");
                        // if (blockCount == numBlocksTotal) {System.out.print("\n");}
                    }
                }

            }

            statusOK = 1;

        } catch (Exception e) {
            statusOK = 0;
            e.printStackTrace();

        } finally {
            try {
                writer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        Map<String, String> info = new HashMap<String, String>();
        info.put("ReadOK", String.valueOf(statusOK));
        info.put("ReadErrors", String.valueOf(blockParser.getErrCounter()));
        info.put("SampleRate", String.valueOf(blockParser.getSampleRate()));

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


    // Check if the accFile has a gyroscope columns
    private static boolean detectGyro(String accFile) {
        boolean hasGyro = false;
        try (FileInputStream accStream = new FileInputStream(accFile);
             FileChannel accChannel = accStream.getChannel()) {

            ByteBuffer block = ByteBuffer.allocate(BLOCKSIZE);
            while (accChannel.read(block) != -1) {
                block.flip();
                block.order(ByteOrder.LITTLE_ENDIAN);
                String header = "" + (char) block.get() + (char) block.get();
                if (header.equals("AX")) {
                    int numAxesBPS = block.get(25) & 0xff;
                    hasGyro = ((numAxesBPS >> 4) & 0x0f) >= 6;
                    break;
                }
                block.clear();
            }
        } catch (IOException e) {
            System.err.println("ERROR: Failed to read the file header.");
            e.printStackTrace();
            System.exit(1);
        }
        return hasGyro;
    }


    private static class BlockParser {

        float sampleRate = -1;
        int errCounter = 0;
        double lastBlockTime = 0;
        LocalDateTime sessionStart = null;
        NpyWriter writer = null;

        public BlockParser(NpyWriter writer) {
            this.writer = writer;
        }

        public float getSampleRate() {
            return sampleRate;
        }

        public int getErrCounter() {
            return errCounter;
        }

        public void parse(ByteBuffer block) {
            block.flip();
            block.order(ByteOrder.LITTLE_ENDIAN);
            String header = (char) block.get() + "";
            header += (char) block.get() + "";

            try {
                if (header.equals("MD")) {
                    /**
                     * TODO: This sometimes raises
                     * java.time.DateTimeException: Invalid value for MonthOfYear (valid values 1 - 12): 0
                     */
                    // sessionStart = getCwaHeaderLoggingStartTime(block);

                    return;

                } else if (header.equals("AX")) {
                    int blockTimeInfo = Math.toIntExact(getUnsignedInt(block, 14));
                    float light = (float) Math.pow(10, (getUnsignedShort(block, 18) & 0x3ff) / 341.0);
                    float temperature = (float) (((getUnsignedShort(block, 20) & 0x3ff) * 150.0 - 20500) / 1000);
                    short rateCode = (short) (block.get(24) & 0xff);
                    short numAxesBPS = (short) (block.get(25) & 0xff);
                    int sampleCount = getUnsignedShort(block, 28);
                    long blockTime = getCwaTimestamp(blockTimeInfo);  // Unix seconds
                    double blockStartTime, blockEndTime;
                    short timestampOffset = 0;
                    float offsetStart = 0;
                    float freq = 0;
                    short checkSum = 0;
                    int numAxes = 0;
		            int accelAxis = -1;
                    int gyroAxis = -1;
					int accelUnit = 256;	// default 1g = 256
					int gyroRange = 2000;	// default 32768 = 2000dps
                    int rawLight = getUnsignedShort(block, 18);

                    accelUnit = 1 << (8 + ((rawLight >>> 13) & 0x07));
					if (((rawLight >> 10) & 0x07) != 0) {
						gyroRange = 8000 / (1 << ((rawLight >>> 10) & 0x07));
					}
                    float gyroUnit = (gyroRange != 0) ? (32768.0f / gyroRange) : 0;

                    // Figure out sample rate (freq)
                    if (rateCode == 0) {
                        // Old format, where pos26 = freq
                        freq = (float) block.getShort(26);
                        offsetStart = 0;
                    } else {
                        // New format
                        timestampOffset = block.getShort(26);
                        freq = 3200.0f / (1 << (15 - (rateCode & 15)));
                        if (freq <= 0) { freq = 1.0f; }
                        offsetStart = (float) -timestampOffset / freq;
                        // Checksum
                        for (int i = 0; i < BLOCKSIZE / 2; i++) { checkSum += block.getShort(i * 2); }
                        if (checkSum != 0) { throw new Exception("Found checksum error. Skipping data block"); }
                    }
                    sampleRate = freq;

                    // Fix so blockTime takes negative offset into account (for <
                    // :00 s boundaries) and so offsetStart is always positive
                    blockTime += (long) Math.floor(offsetStart);
                    offsetStart -= (float) Math.floor(offsetStart);
                    // Start and end of block
                    blockStartTime = (double) blockTime + offsetStart;
                    blockEndTime = blockStartTime + (float) sampleCount / freq;
                    // Fix so packet boundary times are always the same (pushes
                    // error to last packet, would be better to distribute any error
                    // over multiple packets -- would require buffering a few packets)
                    if ((lastBlockTime != 0) && ((blockStartTime - lastBlockTime) < 1.0)) {
                        blockStartTime = lastBlockTime;
                    }
                    lastBlockTime = blockEndTime;

                    // calculate num bytes per sample...
					int bytesPerSample = 0;
					numAxes = (numAxesBPS >> 4) & 0x0f;

					if ((numAxesBPS & 0x0f) == 2) {
                        bytesPerSample = 2 * numAxes;       // 3*16-bit
                    } else if ((numAxesBPS & 0x0f) == 0) {
                        bytesPerSample = 4;                 // 3*10-bit + 2
                    }
					short expectedCount = (short)((bytesPerSample != 0) ? 480 / bytesPerSample : 0);

                    int NUM_AXES_PER_SAMPLE = 3;
                    if ((numAxesBPS & 0x0f) == 2) {
                        bytesPerSample = 6; // 3*16-bit
                    } else if ((numAxesBPS & 0x0f) == 0) {
                        bytesPerSample = 4; // 3*10-bit + 2
                    }

                    numAxes = (numAxesBPS >> 4) & 0x0f;
					if (numAxes >= 6) {
						gyroAxis = 0;
						accelAxis = 3;
					} else if (numAxes >= 3) {
						accelAxis = 0;
					}

                    // Limit values
                    int maxSamples = 480 / bytesPerSample; //80 or 120 samples/block
                    if (sampleCount > maxSamples) { sampleCount = maxSamples; }

                    // Session start?
                    if (sessionStart == null) { sessionStart = getCwaLocalDateTime(blockTimeInfo); }

                    // raw reading values
                    double t = 0;
					short[] sampleValues = new short[sampleCount * numAxes];

                    for (int i = 0; i < sampleCount; i++) {
					    if (bytesPerSample == 4) {
                            long value = getUnsignedInt(block, 30 + 4 * i);
					    	sampleValues[i * numAxes + 0] = (short)((short)(0xffffffc0 & (value <<  6)) >> (6 - ((value >> 30) & 0x03)));	// Sign-extend 10-bit value, adjust for exponent
					    	sampleValues[i * numAxes + 1] = (short)((short)(0xffffffc0 & (value >>  4)) >> (6 - ((value >> 30) & 0x03)));	// Sign-extend 10-bit value, adjust for exponent
					    	sampleValues[i * numAxes + 2] = (short)((short)(0xffffffc0 & (value >> 14)) >> (6 - ((value >> 30) & 0x03)));	// Sign-extend 10-bit value, adjust for exponent
					    } else if (bytesPerSample >= 0) {
					    	for (int j = 0; j < numAxes; j++) {
					    		sampleValues[i * numAxes + j] = block.getShort(30 + (2 * numAxes * i) + (2 * j));
					    	}
					    } else {
					    	for (int j = 0; j < numAxes; j++) {
					    		sampleValues[i * numAxes + j] = 0;
                            }
                        }

                        t = blockStartTime + (double)i * (blockEndTime - blockStartTime) / sampleCount;
                        t *= 1000;  // secs to millis

                        float ax = 0, ay = 0, az = 0;
			            if (accelAxis >= 0) {
			            	ax = (float)sampleValues[numAxes * i + accelAxis + 0] / accelUnit;
			            	ay = (float)sampleValues[numAxes * i + accelAxis + 1] / accelUnit;
			            	az = (float)sampleValues[numAxes * i + accelAxis + 2] / accelUnit;
			            }

			            float gx = 0, gy = 0, gz = 0;
			            if (gyroAxis >= 0) {
			            	gx = (float)sampleValues[numAxes * i + gyroAxis + 0] / gyroUnit;
			            	gy = (float)sampleValues[numAxes * i + gyroAxis + 1] / gyroUnit;
			            	gz = (float)sampleValues[numAxes * i + gyroAxis + 2] / gyroUnit;
			            }

                        if (gyroAxis >= 0) {
                            writer.write(toItems(TimeUnit.MILLISECONDS.toNanos((long) t), ax, ay, az, gx, gy, gz, temperature, light));
                        } else {
                            writer.write(toItems(TimeUnit.MILLISECONDS.toNanos((long) t), ax, ay, az, temperature, light));
                        }

                    }

                    return;

                }

            } catch (Exception e) {
                errCounter++;
                e.printStackTrace();
            }

        }

    }


    private static LocalDateTime getCwaLocalDateTime(int cwaTimeInfo) {
        int year = (int) ((cwaTimeInfo >> 26) & 0x3f) + 2000;
        int month = (int) ((cwaTimeInfo >> 22) & 0x0f);
        int day = (int) ((cwaTimeInfo >> 17) & 0x1f);
        int hours = (int) ((cwaTimeInfo >> 12) & 0x1f);
        int mins = (int) ((cwaTimeInfo >> 6) & 0x3f);
        int secs = (int) ((cwaTimeInfo) & 0x3f);
        LocalDateTime ldt = LocalDateTime.of(year, month, day, hours, mins, secs);
        return ldt;
    }


    private static long getCwaTimestamp(int cwaTimeInfo) {
        LocalDateTime ldt = getCwaLocalDateTime(cwaTimeInfo);
        long timestamp = ldt.toEpochSecond(ZoneOffset.UTC);
        return timestamp;
    }


    private static LocalDateTime getCwaHeaderLoggingStartTime(ByteBuffer block) {
        int delayedLoggingStartTime = (int) getUnsignedInt(block, 13);
        return getCwaLocalDateTime(delayedLoggingStartTime);
    }


    // http://stackoverflow.com/questions/9883472/is-it-possiable-to-have-an-unsigned-bytebuffer-in-java
    private static long getUnsignedInt(ByteBuffer bb, int position) {
        return ((long) bb.getInt(position) & 0xffffffffL);
    }


    // http://stackoverflow.com/questions/9883472/is-it-possiable-to-have-an-unsigned-bytebuffer-in-java
    private static int getUnsignedShort(ByteBuffer bb, int position) {
        return (bb.getShort(position) & 0xffff);
    }


    private static Map<String, Object> toItems(
            long t, float x, float y, float z,
            float temperature, float light) {
        Map<String, Object> items = new HashMap<String, Object>();
        items.put("time", t);
        items.put("x", x);
        items.put("y", y);
        items.put("z", z);
        items.put("temperature", temperature);
        items.put("light", light);
        return items;
    }


    private static Map<String, Object> toItems(
            long t, float x, float y, float z,
            float gyro_x, float gyro_y, float gyro_z,
            float temperature, float light) {
        Map<String, Object> items = new HashMap<String, Object>();
        items.put("time", t);
        items.put("x", x);
        items.put("y", y);
        items.put("z", z);
        items.put("gyro_x", gyro_x);
        items.put("gyro_y", gyro_y);
        items.put("gyro_z", gyro_z);
        items.put("temperature", temperature);
        items.put("light", light);
        return items;
    }


    private static Map<String, Object> toItems(
            long t, double x, double y, double z,
            double temperature, float light) {
        return toItems(
                t, (float) x, (float) y, (float) z,
                (float) temperature, (float) light);
    }


    private static Map<String, Object> toItems(
            long t, double x, double y, double z,
            double gyro_x, double gyro_y, double gyro_z,
            double temperature, float light) {
        return toItems(
                t, (float) x, (float) y, (float) z,
                (float) gyro_x, (float) gyro_y, (float) gyro_z,
                (float) temperature, (float) light);
    }

}
