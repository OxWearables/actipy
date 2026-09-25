# Real-device parser fixtures

These fixtures contain complete native records selected from the large external
recordings. For each available source format, separate windows cover roughly ten
minutes at the beginning, middle, and end of the recording. They exercise the
Java parsers with sustained real device data without adding the full recordings
to the test suite.

`manifest.json` records the source SHA-256, selection, fixture and oracle
hashes, expected schema, sample rate, device identity, and row count.

The matching `*-expected.npz` files are frozen numerical oracles. They were
created by standalone format decoders in
`tests/tools/build_real_parser_fixtures.py`; the builder does not import actipy
or execute the Java readers.

The builder uses these source files from one directory:

- `sample-actigraph.gt3x`
- `sample-actigraph-leap.gt3x`
- `sample-axivity.cwa.gz`
- `sample-geneactiv.bin.gz`

The V1 ActiGraph, Axivity, and GENEActiv recordings are required. The V2
ActiGraph recording is optional when its generated fixtures are already
committed.

It checks the parent of the repository and then the repository-local `data/`
directory by default. A different corpus directory can be supplied explicitly:

```bash
python tests/tools/build_real_parser_fixtures.py --source-dir /path/to/corpus
```

If `sample-actigraph-leap.gt3x` is absent, the already committed ActiGraph V2
fixtures are retained when rebuilding the other formats.

Review changes to the source hashes, selections, and expected arrays before
accepting regenerated files. Normal test runs consume the committed fixtures
directly and do not require access to the external corpus.
