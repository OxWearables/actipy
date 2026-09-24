import java.util.LinkedHashMap;
import java.util.Map;


public class NpyWriterHarness {

    private static final long BASE_TIME = 1_700_000_000_000_000_000L;

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            throw new IllegalArgumentException(
                    "Usage: NpyWriterHarness OUTPUT MODE FLOAT_COLUMNS ROWS");
        }

        String output = args[0];
        String mode = args[1];
        int floatColumns = Integer.parseInt(args[2]);
        int rows = Integer.parseInt(args[3]);

        Map<String, String> schema = schemaFor(mode, floatColumns);
        NpyWriter writer = new NpyWriter(output, schema);
        if ("mutate-schema".equals(mode)) {
            schema.clear();
            schema.put("mutated", "Double");
        }

        for (int row = 0; row < rows; row++) {
            long time = BASE_TIME + row;
            if ("map".equals(mode)) {
                float[] values = valuesFor(row, floatColumns);
                writeMapRow(writer, schema, time, values);
            } else if ("wrong-arity".equals(mode)) {
                int writeColumns = floatColumns == 3 ? 4 : 3;
                writePrimitiveRow(writer, time, valuesFor(row, writeColumns));
            } else {
                writePrimitiveRow(writer, time, valuesFor(row, floatColumns));
            }
        }
        writer.close();
    }

    private static Map<String, String> schemaFor(String mode, int floatColumns) {
        Map<String, String> schema = new LinkedHashMap<String, String>();
        schema.put("time", "wrong-leading-type".equals(mode) ? "Long" : "Datetime");

        if ("map".equals(mode)) {
            for (int column = 0; column < floatColumns; column++) {
                schema.put("f" + column, "Float");
            }
            return schema;
        }

        String[] fieldNames = fieldNamesFor(floatColumns);
        if ("wrong-order".equals(mode)) {
            String first = fieldNames[0];
            fieldNames[0] = fieldNames[1];
            fieldNames[1] = first;
        }
        for (int column = 0; column < fieldNames.length; column++) {
            String type = "wrong-trailing-type".equals(mode)
                    && column == fieldNames.length - 1 ? "Double" : "Float";
            schema.put(fieldNames[column], type);
        }
        return schema;
    }

    private static String[] fieldNamesFor(int floatColumns) {
        switch (floatColumns) {
            case 3:
                return new String[] {"x", "y", "z"};
            case 4:
                return new String[] {"x", "y", "z", "temperature"};
            case 5:
                return new String[] {"x", "y", "z", "temperature", "light"};
            case 8:
                return new String[] {
                        "x", "y", "z", "gyro_x", "gyro_y", "gyro_z",
                        "temperature", "light"};
            default:
                throw new IllegalArgumentException(
                        "Unsupported primitive row width: " + floatColumns);
        }
    }

    private static float[] valuesFor(int row, int floatColumns) {
        float[] values = new float[floatColumns];
        for (int column = 0; column < floatColumns; column++) {
            values[column] = ((row + 1) * (column + 1)) / 8.0f;
        }
        return values;
    }

    private static void writeMapRow(
            NpyWriter writer,
            Map<String, String> schema,
            long time,
            float[] values) throws Exception {
        Map<String, Object> items = new LinkedHashMap<String, Object>();
        items.put("time", time);
        for (int column = 0; column < values.length; column++) {
            items.put("f" + column, values[column]);
        }
        writer.write(items);
    }

    private static void writePrimitiveRow(
            NpyWriter writer,
            long time,
            float[] values) throws Exception {
        switch (values.length) {
            case 3:
                writer.write(time, values[0], values[1], values[2]);
                break;
            case 4:
                writer.write(time, values[0], values[1], values[2], values[3]);
                break;
            case 5:
                writer.write(
                        time, values[0], values[1], values[2], values[3],
                        values[4]);
                break;
            case 8:
                writer.write(
                        time, values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7]);
                break;
            default:
                throw new IllegalArgumentException(
                        "Unsupported primitive row width: " + values.length);
        }
    }
}
