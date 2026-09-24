import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.time.LocalTime;


public class ActigraphReader {

    private static final int INVALID_GT3_FILE = 0;
    private static final int VALID_GT3_V1_FILE = 1;
    private static final int VALID_GT3_V2_FILE = 2;
    private static final int GT3_HEADER_SIZE = 8;
    private static final double MIN_ACCELERATION_SCALE = 16.0;
    private static final double MAX_ACCELERATION_SCALE = 32768.0;
    private static final double MIN_RAW_FULL_SCALE = 1024.0;
    private static final double MAX_RAW_FULL_SCALE = 32768.0;

    // Keep field order aligned with NpyWriter's primitive row layouts.
    private static final Map<String, String> ITEM_NAMES_AND_TYPES;
    static{
        Map<String, String> itemNamesAndTypes = new LinkedHashMap<String, String>();
        itemNamesAndTypes.put("time", "Datetime");
        itemNamesAndTypes.put("x", "Float");
        itemNamesAndTypes.put("y", "Float");
        itemNamesAndTypes.put("z", "Float");
        ITEM_NAMES_AND_TYPES = Collections.unmodifiableMap(itemNamesAndTypes);
    }

    /**
     * Reads a .gt3x file.
     * V1 archives contain activity.bin, lux.bin, and info.txt; V2 archives
     * contain log.bin and info.txt. The method validates the archive, parses
     * its metadata, and processes the corresponding data entry.
     *
     * GT3X timestamps use .NET ticks to represent local time before the
     * reader applies the recorded time-zone offset.
     * TODO: confirm the DST change
     */
    public static void main(String[] args) {

        String accFile = null;
        String outDir = null;
        boolean verbose = false;

        // Parse args string. Example:
        // $ java ActigraphReader -i /path/to/inputFile.bin -o /path/to/outputDir -v
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

        int statusOK = -1;
        double sampleRate = -1;
        int errCounter = 0;  // Number of packet/read errors reported in metadata.
        ZipFile zip = null;
        // Readers for metadata and the version-specific payload entry.
        BufferedReader infoReader = null;
        InputStream activityReader = null;

        String outData = outDir + File.separator + "data.npy";
        NpyWriter writer = new NpyWriter(outData, ITEM_NAMES_AND_TYPES);

        try {
            zip = new ZipFile( new File(accFile), ZipFile.OPEN_READ);

            int gt3Version = getGT3XVersion(zip);
            if (gt3Version == INVALID_GT3_FILE) {
                System.err.println("file " + accFile + " is not a valid V1 or V2 g3tx file");
                statusOK = 0;
            }

            for (Enumeration<?> e = zip.entries(); e.hasMoreElements();) {
                ZipEntry entry = (ZipEntry) e.nextElement();
                if (entry.toString().equals("info.txt")) {
                    infoReader = new BufferedReader(new InputStreamReader(
                            zip.getInputStream(entry), StandardCharsets.UTF_8));
                } else if (entry.toString().equals("activity.bin") && gt3Version == VALID_GT3_V1_FILE) {
                    activityReader = zip.getInputStream(entry);
                } else if (entry.toString().equals("log.bin") && gt3Version == VALID_GT3_V2_FILE) {
                    activityReader = zip.getInputStream(entry);
                }
            }

            double accelerationScale = -1;
            double accelerationMin = Double.NaN, accelerationMax = Double.NaN;
            boolean accelerationScalePresent = false;
            long startDate = -1, stopDate = -1, firstSampleTime=-1;
            String serialNumber = "";
            String infoTimeShift = "00:00:00"; // Treat missing time-zone metadata as UTC.

            while (infoReader.ready()) {
                String line = infoReader.readLine();
                if (line!=null){
                    String[] tokens=line.split(": ");
                    if ((tokens !=null)  && (tokens.length==2)){
                        String key = tokens[0].trim();
                        if (key.startsWith("\uFEFF"))
                            key = key.substring(1).trim();

                        if (key.equals("Sample Rate"))
                            sampleRate=Integer.parseInt(tokens[1].trim());
                        else if (key.equals("Start Date"))
                            firstSampleTime=GT3XfromTickToMillisecond(Long.parseLong(tokens[1].trim()));
                        else if (key.equals("Acceleration Scale")) {
                            accelerationScale=Double.parseDouble(tokens[1].trim());
                            accelerationScalePresent = true;
                        } else if (key.equals("Acceleration Min"))
                            accelerationMin=Double.parseDouble(tokens[1].trim());
                        else if (key.equals("Acceleration Max"))
                            accelerationMax=Double.parseDouble(tokens[1].trim());
                        else if (key.equals("Stop Date"))
                            stopDate=GT3XfromTickToMillisecond(Long.parseLong(tokens[1].trim()));
                        else if (key.equals("Serial Number"))
                            serialNumber=tokens[1].trim();
                        else if (key.equals("TimeZone"))
                            infoTimeShift=tokens[1].trim(); // gt3x calls time shift as time zone
                    }
                }
            }

            // Prefer a plausible scale recorded in info.txt. Older files and
            // invalid metadata can use the known device-family fallback.
            if (!isValidAccelerationScale(accelerationScale)
                    || !isScaleConsistentWithRange(
                            accelerationScale, accelerationMin, accelerationMax)) {
                if (accelerationScalePresent)
                    System.err.println("Ignoring invalid Acceleration Scale in " + accFile);
                accelerationScale = setAccelerationScale(serialNumber);
            }

            if ((sampleRate==-1 || accelerationScale==-1 || firstSampleTime==-1) && gt3Version != VALID_GT3_V2_FILE) {
                System.err.println("Error parsing "+accFile+", info.txt must contain 'Sample Rate', ' Start Date', and (usually) 'Acceleration Scale'.");
                statusOK = 0;
            }

            double sampleDelta = setSampleDelta(sampleRate);

            if (gt3Version == VALID_GT3_V1_FILE) readG3TXv1Pairs(
                    activityReader,
                    infoTimeShift,
                    sampleDelta,
                    sampleRate,
                    accelerationScale,
                    firstSampleTime,
                    writer);
            if (gt3Version == VALID_GT3_V2_FILE) readG3TXv2(
                    activityReader,
                    infoTimeShift,
                    sampleDelta,
                    sampleRate,
                    accelerationScale,
                    writer);

            statusOK = 1;

        } catch (IOException excep) {
            excep.printStackTrace(System.err);
            System.err.println("Error reading/writing file " + accFile + ": " + excep.toString());
            statusOK = 0;

        } finally {
            try {
                zip.close();
                activityReader.close();
                infoReader.close();
                writer.close();
            } catch (Exception e) {
                e.printStackTrace(System.err);
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
     ** Reads x/y/z data from GT3X V2 records in log.bin.
     ** File specification at: https://github.com/actigraph/NHANES-GT3X-File-Format/blob/master/fileformats/activity.bin.md
     ** Data is stored sequentially at the sample rate specified in the header (1/f = sampleDelta in milliseconds)
     ** Each pair of readings occupies an awkward 9 bytes to conserve space, so must be read 2 at a time.
     ** The readings should range from -2046 to 2046, covering -6 to 6 G's,
     ** thus the maximum accuracy is 0.003 G's. The values -2048, -2047 & 2047 should never appear in the stream.
     **/
    private static void readG3TXv2(
            InputStream activityReader,
            String infoTimeShift,
            double sampleDelta,
            double sampleRate,
            double accelerationScale,
            NpyWriter writer
    ) {

        final int PARAMETER_ID = 21;
        final int ACTIVITY_ID = 0;
        final int ACTIVITY2_ID = 26;

        // Two 36-bit XYZ samples occupy nine bytes in the packed V1 stream.
        int checkSum = 0, type=0;
        int i = 0;
        long date = 0;
        int datum;
        int separator = 0;
        int size = 0;
        int initIndex = 0; // Starting index of the current packet.
        boolean isHeader = true;
        int packetCount = 0;

        // Each packet consists of a header, a type-specific payload, and a checksum.
        try {
            while ((datum=activityReader.read())!=-1){
                byte current = (byte)datum;
                if (isHeader) {
                    switch (i-initIndex) {
                        case 0:
                            separator = current;
                            break;
                        case 1:
                            type = current;
                            break;
                        case 2:
                            // Mask before widening so sign extension cannot corrupt the date.
                            date = (long)(current & 0xFF);
                            break;
                        case 3:
                            date = (long)(((current & 0xFF) << 8) ^ date);
                            break;
                        case 4:
                            date = (long)(((current & 0xFF) << 16) ^ date);
                            break;
                        case 5:
                            date = ((long)(current & 0xFF) << 24) ^ date;
                            break;
                        case 6:
                            size = (int)(current & 0xFF);
                            break;
                        case 7:
                            size = (int)(((current & 0xFF) << 8) ^ size);
                    }

                    if (i == initIndex+GT3_HEADER_SIZE-1) {
                        isHeader = false;
                    }

                } else if (isPayload(i, size, initIndex)) {
                    // Packet types have different payload layouts; activity records
                    // are the only data-bearing types currently decoded here.
                    // https://github.com/actigraph/GT3X-File-Format
                    checkSum ^= (byte)current;

                    if (type == PARAMETER_ID) {
                        // Parameter records store key/value pairs in eight-byte groups.
                        byte [] keyPair = new byte[8];
                        byte mydatum;
                        keyPair[0] = current;

                        int k = 1;
                        while (k < 8) {
                            mydatum= (byte) activityReader.read();
                            keyPair[k] = (byte) mydatum;
                            checkSum ^= (byte) mydatum;
                            k++;
                        }

                        // set acceleration scale if present
                        if (isAccelScale(keyPair)) {
                            int keyval = keyPair[4] & 0xFF;
                            keyval = (int)(((keyPair[5] & 0xFF) << 8) ^ keyval);
                            keyval = (int)(((keyPair[6] & 0xFF) << 16) ^ keyval);
                            keyval = (int)(((keyPair[7] & 0xFF) << 24) ^ keyval);
                            double parameterScale = decodePara(keyval);
                            if (isValidAccelerationScale(parameterScale))
                                accelerationScale = parameterScale;
                        }

                        i += 7;
                    } else if (type == ACTIVITY_ID && size > 1) {
                        // A one-byte activity record marks a USB connection event.
                        if (!isValidAccelerationScale(accelerationScale))
                            throw new IllegalStateException("No valid acceleration scale found in GT3X metadata");
                        int [] res = processActivity(
                                infoTimeShift,
                                sampleRate,
                                date,
                                current,
                                i,
                                size,
                                checkSum,
                                initIndex,
                                accelerationScale,
                                activityReader,
                                writer);
                        i = res[0];
                        checkSum = res[1];
                    } else if (type == ACTIVITY2_ID && size > 1) {
                        // A one-byte activity record marks a USB connection event.
                        if (!isValidAccelerationScale(accelerationScale))
                            throw new IllegalStateException("No valid acceleration scale found in GT3X metadata");
                        int [] res = processActivity2(
                                infoTimeShift,
                                sampleRate,
                                date,
                                current,
                                i,
                                size,
                                checkSum,
                                initIndex,
                                accelerationScale,
                                activityReader,
                                writer);
                        i = res[0];
                        checkSum = res[1];
                    }
                } else {
                    checkChecksum(i, separator, type, size, date, checkSum, current);
                    checkSum = 0;
                    date = 0;
                    size = 0;
                    type = 0;
                    separator = 0;
                    // Begin the next packet after validating this packet's checksum.
                    isHeader = true;
                    initIndex = i+1;
                    packetCount++;

                    if (packetCount % 10000 == 0) {
                    }
                }

                i++;
            }
        }
        catch (IOException ex) {
            // End of the GT3X stream.
        }
    }


    /**
     ** Method to read all the x/y/z data from a GT3X (V1) activity.bin file.
     ** File specification at: https://github.com/actigraph/NHANES-GT3X-File-Format/blob/master/fileformats/activity.bin.md
     ** Data is stored sequentially at the sample rate specified in the header (1/f = sampleDelta in milliseconds)
     ** Each pair of readings occupies an awkward 9 bytes to conserve space, so must be read 2 at a time.
     ** The readings should range from -2046 to 2046, covering -6 to 6 G's,
     ** thus the maximum accuracy is 0.003 G's. The values -2048, -2047 & 2047 should never appear in the stream.
     **/
    private static void readG3TXv1Pairs(
            InputStream activityReader,
            String infoTimeShift,
            double sampleDelta,
            double sampleRate,
            double accelerationScale,
            long firstSampleTime, // in milliseconds
            NpyWriter writer
            ) {

        int samples = 0;

        // Two 36-bit XYZ samples occupy nine bytes in the packed V1 stream.
        byte[] bytes=new byte[9];
        int i=0;
        int twoSampleCounter = 0;
        int datum;
        double[] twoSamples = null;

        try {
            while (( datum=activityReader.read())!=-1){
                bytes[i]=(byte)datum;

                if (++i==9){
                    twoSamples = readAccelPair(bytes, accelerationScale);
                    twoSampleCounter = 2;
                }

                while (twoSampleCounter>0) {
                    twoSampleCounter--;
                    i=0;

                    long t = Math.round((1000d*samples)/sampleRate) + firstSampleTime;
                    double x = twoSamples[3-twoSampleCounter*3];
                    double y = twoSamples[4-twoSampleCounter*3];
                    double z = twoSamples[5-twoSampleCounter*3];

                    try {
                        writer.write(
                                TimeUnit.MILLISECONDS.toNanos(t),
                                (float) x, (float) y, (float) z);
                    } catch (NpyWriter.SchemaMismatchException e) {
                        throw e;
                    } catch (Exception e) {
                        System.err.println("Line write error: " + e.toString());
                    }


                    samples += 1;
                }
            }
        }
        catch (IOException ex) {
        }
    }


    private static int [] processActivity(
            String infoTimeShift,
            double sampleRate,
            long firstSampleTime,
            byte current,
            int i,
            int size,
            int checkSum,
            int initIndex,
            double accelerationScale,
            InputStream activityReader,
            NpyWriter writer) {

        double [] sample = new double[3];
        int offset = 0;
        int shifter;
        short axis_val;
        int samples = 0;
        try {
            while (isPayload(i, size, initIndex)) {
                for (int axis = 0; axis < 3; axis++) {
                    if (0 == (offset & 0x07)) {
                        if (i != initIndex + GT3_HEADER_SIZE) {
                            current = (byte) activityReader.read();
                            checkSum ^= (byte) current;
                        }
                        i++;

                        shifter = ((current & 0xFF) << 4);

                        current = (byte) activityReader.read();
                        checkSum ^= (byte) current;
                        i++;

                        shifter |= ((current & 0xF0) >>> 4);
                        offset += 12;
                    } else {
                        shifter = ((current & 0x0F) << 8);

                        current = (byte) activityReader.read();
                        checkSum ^= (byte) current;
                        i++;
                        shifter |= (current & 0xFF);
                        offset += 12;
                    }
                    if (shifter > 2047)
                        shifter += 61440;

                    axis_val = (short) shifter;
                    sample[axis] = axis_val / accelerationScale;
                    sample[axis] = (double) Math.round(sample[axis] * 1000d) / 1000d;
                }
                long myTime = Math.round((1000d*samples)/sampleRate) + firstSampleTime*1000;
                samples += 1;


                // V1 stores Y before X; restore the public X/Y/Z order here.
                try {
                    writer.write(
                            TimeUnit.MILLISECONDS.toNanos(myTime),
                            (float) sample[1], (float) sample[0], (float) sample[2]);
                } catch (NpyWriter.SchemaMismatchException e) {
                    throw e;
                } catch (Exception e) {
                    System.err.println("Line write error: " + e.toString());
                }

            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
            System.err.println("error when reading activity at byte " + i + ": " + ex.toString());
            System.exit(-2);
        }

        return new int[] {i, checkSum};
    }


    private static long getTrueUnixTime(long myTime, String infoTimeShift) {
        int shiftSign = 1;
        if (infoTimeShift.charAt(0) == '-') {
            shiftSign = -1;
            infoTimeShift = infoTimeShift.substring(1);
        }

        LocalTime timeShift = LocalTime.parse(infoTimeShift);
        long timeShiftMilli = 1000 * (shiftSign * timeShift.getHour() * 60 * 60 +
                timeShift.getMinute() * 60); // Time shift relative to UTC.
        return myTime - timeShiftMilli;
    }


    private static int [] processActivity2(
            String infoTimeShift,
            double sampleRate,
            long firstSampleTime,
            byte current,
            int i,
            int size,
            int checkSum,
            int initIndex,
            double accelerationScale,
            InputStream activityReader,
            NpyWriter writer) {

        double [] sample = new double[3];
        int shifter;
        short axis_val;
        int samples = 0;
        try {
            while (isPayload(i, size, initIndex)) {
                for (int axis = 0; axis < 3; axis++) {
                    if (i != initIndex + GT3_HEADER_SIZE) {
                        current = (byte) activityReader.read();
                        checkSum ^= (byte) current;
                    }

                    shifter = current & 0xff;
                    current = (byte) activityReader.read();
                    checkSum ^= (byte) current;
                    shifter |= ((current & 0xff) << 8);
                    i += 2;

                    axis_val = (short) shifter;

                    sample[axis] = axis_val / accelerationScale;
                    sample[axis] = (double) Math.round(sample[axis] * 1000d) / 1000d;
                }

                long myTime = Math.round((1000d*samples)/sampleRate) + firstSampleTime*1000;
                samples += 1;

                try {
                    writer.write(
                            TimeUnit.MILLISECONDS.toNanos(myTime),
                            (float) sample[0], (float) sample[1], (float) sample[2]);
                } catch (NpyWriter.SchemaMismatchException e) {
                    throw e;
                } catch (Exception e) {
                    System.err.println("Line write error: " + e.toString());
                }

            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
            System.err.println("error when reading activity at byte " + i + ": " + ex.toString());
            System.exit(-2);
        }

        return new int[] {i, checkSum};
    }


    private static double[] readAccelPair(byte[] bytes, double accelerationScale) {

        int datum = 0;
        datum=(bytes[0]&0xff);datum=datum<<4;datum|=(bytes[1]&0xff)>>>4;
        short y1=(short)datum;
        if (y1>2047)
            y1+=61440;

        datum=bytes[1]&0x0F;datum=datum<<8;datum|=(bytes[2]&0xff);
        short x1=(short)datum;
        if (x1>2047)
            x1+=61440;

        datum=bytes[3]&0xff;datum=datum<<4;datum|=(bytes[4]&0xff)>>>4;
        short z1=(short)datum;
        if (z1>2047)
            z1+=61440;

        datum=bytes[4]&0x0F;datum=datum<<8;datum|=(bytes[5]&0xff);
        short y2=(short) datum;
        if (y2>2047)
            y2+=61440;

        datum=(bytes[6]&0xff);datum=datum<<4;datum|=(bytes[7]&0xff)>>>4;
        short x2=(short)datum;
        if (x2>2047)
            x2+=61440;

        datum=bytes[7]&0x0F;datum=datum<<8;datum|=(bytes[8]&0xff);
        short z2=(short)datum;
        if (z2>2047)
            z2+=61440;

        // Convert raw axis values to g using the metadata scale.
        double gx1=x1/accelerationScale;
        double gy1=y1/accelerationScale;
        double gz1=z1/accelerationScale;

        double gx2=x2/accelerationScale;
        double gy2=y2/accelerationScale;
        double gz2=z2/accelerationScale;

        return new double[] {gx1, gy1, gz1, gx2, gy2, gz2};
    }


    /**
     * Checks the checksum formed from the packet payload and header fields.
     */
    private static void checkChecksum(
            int i,
            int separator,
            int type,
            int size,
            long date,
            int checkSum,
            int target_value) {

        checkSum ^= (byte)separator;
        checkSum ^= (byte)type;
        checkSum ^= (byte)(size & 0xFF);
        checkSum ^= (byte)((size >> 8) & 0xFF);
        checkSum ^= (byte)(date & 0xFF);
        checkSum ^= (byte)((date >> 8) & 0xFF);
        checkSum ^= (byte)((date >> 16) & 0xFF);
        checkSum ^= (byte)((date >> 24) & 0xFF);

        // The stored checksum is the one's complement of the accumulated bytes.
        checkSum = (byte)~checkSum;
        if (checkSum != target_value) {
            System.exit(-1);
        }
    }


    private static double setAccelerationScale(String serialNumber) {
        double ACCELERATION_SCALE_FACTOR_NEO_CLE = 341.0; // 2046 raw units over 6 g.
        double ACCELERATION_SCALE_FACTOR_MOS = 256.0; // 2048 raw units over 8 g.
        double accelerationScale = -1;

        if((serialNumber.startsWith("NEO") || (serialNumber.startsWith("CLE")))) {
            accelerationScale = ACCELERATION_SCALE_FACTOR_NEO_CLE;
        } else if(serialNumber.startsWith("MOS")){
            accelerationScale = ACCELERATION_SCALE_FACTOR_MOS;
        }
        return accelerationScale;
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
        if (Double.isNaN(accelerationMin) || Double.isNaN(accelerationMax))
            return true;
        if (!Double.isFinite(accelerationMin)
                || !Double.isFinite(accelerationMax)
                || accelerationMin >= 0
                || accelerationMax <= 0)
            return false;

        double range = Math.max(Math.abs(accelerationMin), Math.abs(accelerationMax));
        double rawFullScale = accelerationScale * range;
        return rawFullScale >= MIN_RAW_FULL_SCALE
                && rawFullScale <= MAX_RAW_FULL_SCALE;
    }


    private static double setSampleDelta(double sampleRate) {
        double sampleDelta = 1000.0/sampleRate;

        return sampleDelta;
    }


    /**
     ** Returns whether the byte index lies within the packet payload.
     **
     */
    private static boolean isPayload(int i, int size, int initIndex) {
        if (i >= initIndex+GT3_HEADER_SIZE && i < (initIndex+GT3_HEADER_SIZE+size)) return true;
        else return false;
    }

    /**
     * This was translated into Java from
     * https://github.com/actigraph/GT3X-File-Format/blob/master/LogRecords/Parameters.md
     */
    private static double decodePara(int value) {
        final double FLOAT_MAXIMUM = 8388608.0;                  /* 2^23  */
        final int ENCODED_MINIMUM = 0x00800000;
        final int ENCODED_MAXIMUM = 0x007FFFFF;
        final int SIGNIFICAND_MASK = 0x00FFFFFF;
        final int EXPONENT_MASK = 0xFF000000;
        final int EXPONENT_OFFSET = 24;

        double significand;
        int exponent;
        int i32;

        if (ENCODED_MAXIMUM == value)
            return Integer.MAX_VALUE;
        else if (ENCODED_MAXIMUM == value)
            return -Integer.MAX_VALUE;

        i32 = (int) ((value & EXPONENT_MASK) >>> EXPONENT_OFFSET);
        if (0 != (i32 & 0x80))
            i32 = (int)((int)i32 | 0xFFFFFF00);
        exponent = (int)i32;

        i32 = (int)(value & SIGNIFICAND_MASK);
        if (0 != (i32 & ENCODED_MINIMUM))
            i32 = (int)((int)i32 | 0xFF000000);

        significand = (double) i32 / FLOAT_MAXIMUM;

        return significand * Math.pow(2.0, exponent);
    }


    private static boolean isAccelScale(byte[] keyPairs) {
        int addressSpace = keyPairs[0];
        int identifier = keyPairs[2];
        if (addressSpace == 0 && identifier == 55) return true;
        else return false;
    }


    /**
     ** Converts the .NET ticks used by Actigraph GT3X to local milliseconds.
     ** Based on: https://github.com/SPADES-PUBLIC/mHealth-GT3X-converter-public/blob/master/src/com/qmedic/data/converter/gt3x/GT3XUtils.java
     *
     * A .NET tick represents 100 nanoseconds.
     * https://docs.microsoft.com/en-us/dotnet/api/system.datetime.ticks?view=netcore-3.1
     **/
    private static long GT3XfromTickToMillisecond(final long ticks)
    {
        Date date = new Date((ticks - 621355968000000000L) / 10000);
        return date.getTime();
    }


    /*
     * This method checks which GT3 version the zipfile contains.
     * Return 1 for v1, 2 for v2, 0 for invalid GT3 file
     */
    private static int getGT3XVersion(final ZipFile zip) throws IOException {

        // Check for the entries required by each supported GT3X version.
        boolean hasActivityData = false;
        boolean hasLuxData = false;
        boolean hasInfoData = false;
        boolean hasLogData = false;
        for (Enumeration<?> e = zip.entries(); e.hasMoreElements();) {
            ZipEntry entry = (ZipEntry) e.nextElement();
            if (entry.toString().equals("activity.bin"))
                hasActivityData = true;
            if (entry.toString().equals("lux.bin"))
                hasLuxData = true;
            if (entry.toString().equals("info.txt"))
                hasInfoData = true;
            if (entry.toString().equals("log.bin"))
                hasLogData = true;
        }

        if (hasActivityData && hasLuxData && hasInfoData) {
            return VALID_GT3_V1_FILE;
        } else if (hasInfoData && hasLogData) {
            return VALID_GT3_V2_FILE;
        }

        return INVALID_GT3_FILE;
    }
}
