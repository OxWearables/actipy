import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.GZIPOutputStream;


public class NpyWriter implements AutoCloseable {

    private static final int ROWS_PER_BUFFER = 8192;
    private static final ByteOrder NATIVE_BYTE_ORDER = ByteOrder.nativeOrder();
    private static final char NUMPY_BYTE_ORDER =
            NATIVE_BYTE_ORDER == ByteOrder.BIG_ENDIAN ? '>' : '<';
    private static final byte NPY_MAJ_VERSION = 1;
    private static final byte NPY_MIN_VERSION = 0;
    private static final int BLOCK_SIZE = 16;
    private static final int HEADER_SIZE = BLOCK_SIZE * 16;
    private static final byte[] NPY_HEADER = new byte[] {
            (byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};

    public enum Layout {
        XYZ(new String[] {"x", "y", "z"}),
        XYZT(new String[] {"x", "y", "z", "temperature"}),
        XYZTL(new String[] {"x", "y", "z", "temperature", "light"}),
        XYZ_GYRO_TL(new String[] {
                "x", "y", "z", "gyro_x", "gyro_y", "gyro_z",
                "temperature", "light"});

        private final String[] floatFields;

        Layout(String[] floatFields) {
            this.floatFields = floatFields;
        }

        private Map<String, String> schema() {
            Map<String, String> schema = new LinkedHashMap<String, String>();
            schema.put("time", "Datetime");
            for (String field : floatFields) {
                schema.put(field, "Float");
            }
            return schema;
        }
    }

    private enum FieldType {
        INTEGER("Integer", Integer.BYTES, "i4", Integer.class) {
            void put(ByteBuffer buffer, Object value) {
                buffer.putInt((Integer) value);
            }
        },
        SHORT("Short", Short.BYTES, "i2", Short.class) {
            void put(ByteBuffer buffer, Object value) {
                buffer.putShort((Short) value);
            }
        },
        LONG("Long", Long.BYTES, "i8", Long.class) {
            void put(ByteBuffer buffer, Object value) {
                buffer.putLong((Long) value);
            }
        },
        FLOAT("Float", Float.BYTES, "f4", Float.class) {
            void put(ByteBuffer buffer, Object value) {
                buffer.putFloat((Float) value);
            }
        },
        DOUBLE("Double", Double.BYTES, "f8", Double.class) {
            void put(ByteBuffer buffer, Object value) {
                buffer.putDouble((Double) value);
            }
        },
        DATETIME("Datetime", Long.BYTES, "M8[ns]", Long.class) {
            void put(ByteBuffer buffer, Object value) {
                buffer.putLong((Long) value);
            }
        };

        private final String tag;
        private final int byteWidth;
        private final String numpyType;
        private final Class<?> valueClass;

        FieldType(String tag, int byteWidth, String numpyType, Class<?> valueClass) {
            this.tag = tag;
            this.byteWidth = byteWidth;
            this.numpyType = numpyType;
            this.valueClass = valueClass;
        }

        abstract void put(ByteBuffer buffer, Object value);

        private boolean accepts(Object value) {
            return value != null && valueClass.isInstance(value);
        }

        private String numpyDescriptor() {
            return NUMPY_BYTE_ORDER + numpyType;
        }

        private static FieldType fromTag(String tag) {
            for (FieldType type : values()) {
                if (type.tag.equals(tag)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("Unrecognized item type: " + tag);
        }
    }

    public static class SchemaMismatchException extends IllegalStateException {
        private static final long serialVersionUID = 1L;

        SchemaMismatchException(String message) {
            super(message);
        }
    }

    private final String outputFile;
    private final Map<String, FieldType> fields;
    private final Layout primitiveLayout;
    private final ByteBuffer buffer;
    private final File file;
    private final RandomAccessFile randomAccessFile;
    private int linesWritten;
    private boolean closed;

    public NpyWriter(String outputFile, Layout layout) {
        this(outputFile, layout.schema());
    }

    public NpyWriter(String outputFile, Map<String, String> itemNamesAndTypes) {
        if (itemNamesAndTypes == null || itemNamesAndTypes.isEmpty()) {
            throw new IllegalArgumentException("The .npy schema must not be empty");
        }

        this.outputFile = outputFile;
        this.fields = parseFields(itemNamesAndTypes);
        this.primitiveLayout = getPrimitiveLayout(fields);
        this.buffer = ByteBuffer.allocate(
                ROWS_PER_BUFFER * getBytesPerLine(fields)).order(NATIVE_BYTE_ORDER);
        this.file = new File(outputFile);

        RandomAccessFile openedFile = null;
        try {
            openedFile = new RandomAccessFile(file, "rw");
            openedFile.setLength(0);
            reserveHeader(openedFile);
        } catch (IOException error) {
            if (openedFile != null) {
                try {
                    openedFile.close();
                } catch (IOException closeError) {
                    error.addSuppressed(closeError);
                }
            }
            throw new UncheckedIOException(
                    "The .npy file " + outputFile + " could not be created", error);
        }
        this.randomAccessFile = openedFile;
    }

    public NpyWriter(String outputFile) {
        this(outputFile, Layout.XYZ);
    }

    public void write(Map<String, Object> items) throws IOException {
        ensureOpen();
        validateItems(items);
        for (Map.Entry<String, FieldType> field : fields.entrySet()) {
            field.getValue().put(buffer, items.get(field.getKey()));
        }
        finishRow();
    }

    public void write(long time, float x, float y, float z) throws IOException {
        ensureOpen();
        requirePrimitiveLayout(Layout.XYZ);
        buffer.putLong(time);
        buffer.putFloat(x);
        buffer.putFloat(y);
        buffer.putFloat(z);
        finishRow();
    }

    public void write(
            long time,
            float x, float y, float z, float temperature) throws IOException {
        ensureOpen();
        requirePrimitiveLayout(Layout.XYZT);
        buffer.putLong(time);
        buffer.putFloat(x);
        buffer.putFloat(y);
        buffer.putFloat(z);
        buffer.putFloat(temperature);
        finishRow();
    }

    public void write(
            long time,
            float x, float y, float z, float temperature,
            float light) throws IOException {
        ensureOpen();
        requirePrimitiveLayout(Layout.XYZTL);
        buffer.putLong(time);
        buffer.putFloat(x);
        buffer.putFloat(y);
        buffer.putFloat(z);
        buffer.putFloat(temperature);
        buffer.putFloat(light);
        finishRow();
    }

    public void write(
            long time,
            float x, float y, float z, float gyroX,
            float gyroY, float gyroZ, float temperature, float light) throws IOException {
        ensureOpen();
        requirePrimitiveLayout(Layout.XYZ_GYRO_TL);
        buffer.putLong(time);
        buffer.putFloat(x);
        buffer.putFloat(y);
        buffer.putFloat(z);
        buffer.putFloat(gyroX);
        buffer.putFloat(gyroY);
        buffer.putFloat(gyroZ);
        buffer.putFloat(temperature);
        buffer.putFloat(light);
        finishRow();
    }

    private void finishRow() throws IOException {
        linesWritten++;
        if (!buffer.hasRemaining()) {
            flushBuffer();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Cannot write to a closed NpyWriter");
        }
    }

    private void requirePrimitiveLayout(Layout expected) {
        if (primitiveLayout != expected) {
            throw new SchemaMismatchException(
                    "Primitive row layout " + expected + " does not match schema");
        }
    }

    private void validateItems(Map<String, Object> items) {
        if (items == null) {
            throw new IllegalArgumentException("Row items must not be null");
        }
        for (Map.Entry<String, FieldType> field : fields.entrySet()) {
            Object value = items.get(field.getKey());
            if (!items.containsKey(field.getKey())) {
                throw new IllegalArgumentException(
                        "Missing value for field: " + field.getKey());
            }
            if (!field.getValue().accepts(value)) {
                throw new IllegalArgumentException(
                        "Value for field " + field.getKey()
                        + " must be " + field.getValue().valueClass.getSimpleName());
            }
        }
    }

    private static void reserveHeader(RandomAccessFile output) throws IOException {
        int headerPrefixSize = NPY_HEADER.length + 2 + Short.BYTES;
        output.write(new byte[headerPrefixSize + HEADER_SIZE]);
    }

    private void flushBuffer() throws IOException {
        int bytesUsed = buffer.position();
        if (bytesUsed > 0) {
            randomAccessFile.write(buffer.array(), 0, bytesUsed);
            buffer.clear();
        }
    }

    private void writeHeader() throws IOException {
        randomAccessFile.seek(0);
        randomAccessFile.write(NPY_HEADER);
        randomAccessFile.write(NPY_MAJ_VERSION);
        randomAccessFile.write(NPY_MIN_VERSION);

        StringBuilder dataHeader = new StringBuilder("{ 'descr': [");
        int fieldIndex = 0;
        for (Map.Entry<String, FieldType> field : fields.entrySet()) {
            if (fieldIndex > 0) {
                dataHeader.append(',');
            }
            dataHeader.append("('")
                    .append(field.getKey())
                    .append("','")
                    .append(field.getValue().numpyDescriptor())
                    .append("')");
            fieldIndex++;
        }
        dataHeader.append("]")
                .append(", 'fortran_order': False")
                .append(", 'shape': (")
                .append(linesWritten)
                .append(",), }");

        int headerLength = dataHeader.length() + 1;
        if (headerLength > HEADER_SIZE) {
            throw new IOException("The .npy header is too large");
        }
        while (dataHeader.length() < HEADER_SIZE - 1) {
            dataHeader.append(' ');
        }
        dataHeader.append('\n');

        writeLittleEndianShort(randomAccessFile, (short) HEADER_SIZE);
        randomAccessFile.writeBytes(dataHeader.toString());
        randomAccessFile.seek(randomAccessFile.length());
    }

    private void finalizeFile() throws IOException {
        flushBuffer();
        writeHeader();
    }

    public void compress(String compressedOutputFile) {
        ensureOpen();
        try {
            File compressedFile = new File(compressedOutputFile);
            if (file.getCanonicalFile().equals(compressedFile.getCanonicalFile())) {
                throw new IllegalArgumentException(
                        "Compressed output must differ from " + outputFile);
            }

            finalizeFile();
            try (FileInputStream input = new FileInputStream(file);
                 GZIPOutputStream output = new GZIPOutputStream(
                         new FileOutputStream(compressedFile))) {
                byte[] compressedBuffer = new byte[8192];
                int length;
                while ((length = input.read(compressedBuffer)) != -1) {
                    output.write(compressedBuffer, 0, length);
                }
            }
        } catch (IOException error) {
            throw new UncheckedIOException("Could not compress " + outputFile, error);
        }
    }

    public void compress() {
        compress(outputFile + ".gz");
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }

        IOException failure = null;
        try {
            finalizeFile();
        } catch (IOException error) {
            failure = error;
        }
        try {
            randomAccessFile.close();
        } catch (IOException error) {
            if (failure == null) {
                failure = error;
            } else {
                failure.addSuppressed(error);
            }
        } finally {
            closed = true;
        }

        if (failure != null) {
            throw new UncheckedIOException("Could not finalize " + outputFile, failure);
        }
    }

    public void closeAndDelete() {
        close();
        if (file.exists() && !file.delete()) {
            throw new IllegalStateException("Could not delete " + outputFile);
        }
    }

    private static Map<String, FieldType> parseFields(
            Map<String, String> itemNamesAndTypes) {
        Map<String, FieldType> parsed = new LinkedHashMap<String, FieldType>();
        for (Map.Entry<String, String> field : itemNamesAndTypes.entrySet()) {
            parsed.put(field.getKey(), FieldType.fromTag(field.getValue()));
        }
        return Collections.unmodifiableMap(parsed);
    }

    private static int getBytesPerLine(Map<String, FieldType> fields) {
        int bytesPerLine = 0;
        for (FieldType type : fields.values()) {
            bytesPerLine += type.byteWidth;
        }
        return bytesPerLine;
    }

    private static Layout getPrimitiveLayout(Map<String, FieldType> fields) {
        for (Layout layout : Layout.values()) {
            if (matchesPrimitiveLayout(fields, layout.schema())) {
                return layout;
            }
        }
        return null;
    }

    private static boolean matchesPrimitiveLayout(
            Map<String, FieldType> fields,
            Map<String, String> expectedSchema) {
        if (fields.size() != expectedSchema.size()) {
            return false;
        }

        java.util.Iterator<Map.Entry<String, FieldType>> actual =
                fields.entrySet().iterator();
        java.util.Iterator<Map.Entry<String, String>> expected =
                expectedSchema.entrySet().iterator();
        while (actual.hasNext()) {
            Map.Entry<String, FieldType> actualField = actual.next();
            Map.Entry<String, String> expectedField = expected.next();
            if (!actualField.getKey().equals(expectedField.getKey())
                    || actualField.getValue() != FieldType.fromTag(expectedField.getValue())) {
                return false;
            }
        }
        return true;
    }

    private static void writeLittleEndianShort(
            RandomAccessFile output,
            short value) throws IOException {
        output.writeByte(value & 0xFF);
        output.writeByte((value >>> 8) & 0xFF);
    }
}
