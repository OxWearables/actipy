import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;


final class ReaderSupport {

    private ReaderSupport() {
    }

    static final class Options {
        final String inputFile;
        final String outputDirectory;
        final boolean verbose;

        private Options(String inputFile, String outputDirectory, boolean verbose) {
            this.inputFile = inputFile;
            this.outputDirectory = outputDirectory;
            this.verbose = verbose;
        }

        static Options parse(String[] args) {
            String inputFile = null;
            String outputDirectory = null;
            boolean verbose = false;

            for (int index = 0; index < args.length; index++) {
                if ("-i".equals(args[index]) && index < args.length - 1) {
                    inputFile = args[++index];
                } else if ("-o".equals(args[index]) && index < args.length - 1) {
                    outputDirectory = args[++index];
                } else if ("-v".equals(args[index])) {
                    verbose = true;
                }
            }

            if (inputFile == null) {
                throw new IllegalArgumentException("No input file specified");
            }
            if (outputDirectory == null) {
                throw new IllegalArgumentException("No output directory specified");
            }
            return new Options(inputFile, outputDirectory, verbose);
        }

        String dataPath() {
            return outputDirectory + File.separator + "data.npy";
        }
    }

    static final class Result {
        int readOk;
        int readErrors;
        double sampleRate = -1;

        void recordRecoverableError(String message) {
            readErrors++;
            System.err.println(message);
        }
    }

    static class FormatException extends Exception {
        private static final long serialVersionUID = 1L;

        FormatException(String message) {
            super(message);
        }

        FormatException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    interface Converter {
        void convert(Options options, Result result) throws Exception;
    }

    static void run(String[] args, Converter converter) {
        Options options;
        try {
            options = Options.parse(args);
        } catch (IllegalArgumentException error) {
            System.err.println("ERROR: " + error.getMessage());
            System.exit(1);
            return;
        }

        Result result = new Result();
        int exitCode = 0;
        try {
            Files.deleteIfExists(new File(options.dataPath()).toPath());
            converter.convert(options, result);
            result.readOk = 1;
        } catch (Exception error) {
            result.readOk = 0;
            result.readErrors++;
            System.err.println("Error reading " + options.inputFile + ": "
                    + error.getMessage());
            error.printStackTrace(System.err);
            exitCode = 1;
        }

        try {
            writeInfo(options.outputDirectory, result);
        } catch (IOException error) {
            System.err.println("Error writing reader metadata: " + error.getMessage());
            error.printStackTrace(System.err);
            exitCode = 1;
        }

        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    static void writeInfo(String outputDirectory, Result result) throws IOException {
        String outputPath = outputDirectory + File.separator + "info.txt";
        try (FileWriter writer = new FileWriter(outputPath)) {
            writer.write("ReadOK:" + result.readOk + "\n");
            writer.write("ReadErrors:" + result.readErrors + "\n");
            writer.write("SampleRate:" + result.sampleRate + "\n");
        }
    }
}
