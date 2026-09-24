# Real-device parser fixtures

These fixtures are small, complete record selections from the large recordings
under the repository-local `data/` directory. They exercise the Java parsers
with real device bytes without adding the full recordings to the test suite.

`manifest.json` records the source SHA-256, selection, fixture and oracle
hashes, expected schema, sample rate, device identity, and row count.

The matching `*-expected.npz` files are frozen numerical oracles. They were
created by standalone format decoders in
`tests/tools/build_real_parser_fixtures.py`; the builder does not import actipy
or execute the Java readers.

To rebuild the fixtures when all four source recordings are available:

```bash
python tests/tools/build_real_parser_fixtures.py
```

Review changes to the source hashes, selections, and expected arrays before
accepting regenerated files. Normal test runs consume the committed fixtures
