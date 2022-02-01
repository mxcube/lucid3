# /*##########################################################################
# Copyright (C) 2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
"""
Lucid 3 project - core module
"""

__author__ = "Olof Svensson"
__contact__ = "svensson@esrf.eu"
__copyright__ = "ESRF, 2021"
__updated__ = "2022-02-01"

import pathlib

import lucid3

REFERENCE_DICT = {
    "snapshot_1.jpeg": {"xpos": 675, "ypos": 494},
    "snapshot_2.jpeg": {"xpos": 676, "ypos": 482},
    "snapshot_3.jpeg": {"xpos": 665, "ypos": 508},
    "snapshot_4.jpeg": {"xpos": 674, "ypos": 515},
}


def test_lucid3():
    test_data_path = pathlib.Path(__file__).parent / "data"
    for file_name, reference in REFERENCE_DICT.items():
        file_path = str(test_data_path / file_name)
        result = lucid3.find_loop(file_path)
        assert result[0] == "Coord"
        assert result[1] == reference["xpos"]
        assert result[2] == reference["ypos"]
