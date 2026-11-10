This is a simple script that makes geometry cartoons from geometry descriptions.

To make the standalone MAIA nozzle profile (PNG and PDF):

```sh
python3 make_maia_nozzle.py
```

The plot reads its dimensions directly from the MAIA DD4hep compact XML and
uses millimeters by default. Use `--side right`, `--unit cm`, or
`--output-dir PATH` to change the defaults.
