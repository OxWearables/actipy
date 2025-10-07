## Glossary

This document describes all fields that may appear in the `info` dictionary returned by `read_device()` and `process()` functions.

### File and Device Information

- **Filename**: The name of the data file.
- **Filesize(MB)**: The size of the file in megabytes.
- **Device**: The brand or model of the device that recorded the data (e.g., "Axivity", "Actigraph", "GENEActiv", "Matrix").
- **DeviceID**: A unique identifier for the specific device used.

### Data Reading

- **ReadOK**: A binary indicator (1 for success, 0 for failure) showing whether the data was read successfully.
- **ReadErrors**: Number of errors encountered while reading data from the device.
- **SampleRate**: The frequency at which data points are sampled per second, measured in Hz.

### Data Quality and Coverage

- **StartTime**: Timestamp marking the beginning of the data recording (format: "YYYY-MM-DD HH:MM:SS").
- **EndTime**: Timestamp marking the end of the data recording (format: "YYYY-MM-DD HH:MM:SS").
- **NumTicks**: Total number of data points recorded.
- **WearTime(days)**: Total time the device was worn, expressed in days. This is the sum of all valid (non-NaN) data segments with gaps smaller than 1 second. This value is updated if nonwear detection is performed to exclude segments flagged as nonwear.
- **DataSpan(days)**: Time span of the data (difference between last and first timestamps), expressed in days.
- **NumInterrupts**: Number of interruptions in the data recording (gaps larger than 1 second between consecutive samples). This value is updated after nonwear detection - nonwear episodes are counted as additional interrupts.
- **Covers24hOK**: A binary indicator (1 for yes, 0 for no) showing whether the data covers all 24 hours of the day with at least 1% coverage per hour. This value is updated after nonwear detection to reflect coverage excluding nonwear periods.

### Lowpass Filtering

- **LowpassOK**: A binary indicator (1 for success, 0 for failure) showing whether the lowpass filter was applied successfully.
- **LowpassCutoff(Hz)**: The cutoff frequency in Hertz for the lowpass filter.

### Gravity Calibration

- **CalibOK**: A binary indicator (1 for successful calibration, 0 for unsuccessful) showing the calibration status.
- **CalibErrorBefore(mg)**: Calibration error before any correction was applied, measured in milli-g (mean absolute error).
- **CalibErrorAfter(mg)**: Calibration error after attempting correction, measured in milli-g (mean absolute error).
- **CalibNumSamples**: Number of stationary samples used for calibration.
- **CalibNumIters**: Number of iterations performed during calibration optimization.
- **CalibxIntercept**: X-axis intercept coefficient from the calibration transform.
- **CalibyIntercept**: Y-axis intercept coefficient from the calibration transform.
- **CalibzIntercept**: Z-axis intercept coefficient from the calibration transform.
- **CalibxSlope**: X-axis slope coefficient from the calibration transform.
- **CalibySlope**: Y-axis slope coefficient from the calibration transform.
- **CalibzSlope**: Z-axis slope coefficient from the calibration transform.
- **CalibxSlopeT**: X-axis temperature coefficient from the calibration transform (only present if temperature data available).
- **CalibySlopeT**: Y-axis temperature coefficient from the calibration transform (only present if temperature data available).
- **CalibzSlopeT**: Z-axis temperature coefficient from the calibration transform (only present if temperature data available).

### Nonwear Detection

- **NonwearTime(days)**: Total time the device was not worn, expressed in days.
- **NumNonwearEpisodes**: Number of separate episodes when the device was not worn.

### Resampling

- **ResampleRate**: The new sampling rate after data has been resampled, measured in Hz.
- **NumTicksAfterResample**: Number of data points after resampling.
- **FirstCompleteMinuteStart**: Timestamp of the first complete minute used as the starting point (format: "YYYY-MM-DD HH:MM:SS"). Only present if `start_first_complete_minute=True`.
