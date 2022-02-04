lucid3: Loop and uCrystals identification version 3
===================================================

Authors: Olof Svensson and Sogeti

lucid3 is a computer vision Python library that detects
crystallography samples holders.

mxCuBE relies lucid3 to provide automatic centring.

Dependencies:
- python >= 3.5
- opencv >= 2.4

## Development

Install from source

```bash
python -m pip install -e .[dev]
```

Run tests

```bash
pytest
```

Launch lucid3 from the command line

```
lucid3 -h
usage: lucid3 [-h] [--vertical] [--display] [--create_result_file] [--result_directory RESULT_DIRECTORY] [--debug] snapshot_file_path

Application for finding loop in diffractometer snapshot of sample

required arguments:
  snapshot_file_path    Path to snapshot jpg/png file

optional arguments:
  --vertical            Vertical rotation axis
  --display             Display snapshot with result
  --create_result_file  Create an image file with result
  --result_directory RESULT_DIRECTORY
                        Directory to where store result file (default cwd)
  --debug               Display snapshot with all intermediate steps and with result
```
