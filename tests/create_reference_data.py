# coding: utf-8
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
Lucid 3 project - code for creating test data
"""

__author__ = "Olof Svensson"
__contact__ = "svensson@esrf.eu"
__copyright__ = "ESRF, 2017"
__updated__ = "2018-08-20"

import os
import glob
import shutil
import hashlib
import scipy.misc
import matplotlib.pyplot as plt

class ImagePosition(object):
    xPos = None
    yPos = None

def onclick(event):
    if event.xdata != None and event.ydata != None:
        print(event.xdata, event.ydata)
        ImagePosition.xPos = event.xdata
        ImagePosition.yPos = event.ydata

# From https://stackoverflow.com/a/3431838:
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

snapshotDir = "/data/id30a3/inhouse/opid30a3/snapshots/"
savePath = "/scisoft/pxsoft/data/lucid/reference/id30a3"
if not os.path.exists(savePath):
    os.makedirs(savePath, 0755)

# Create a list of already marked images (if exist)
listMarked = glob.glob(os.path.join(savePath, "*"))

for filePath in glob.glob(os.path.join(snapshotDir, "*.png")):
    ImagePosition.xPos = None
    ImagePosition.yPos = None
    # Check if marked image exists
    md5sum = md5(filePath)
    md5sumFirst10 = md5sum[0:10]
    marked = False
    if "marked" in filePath:
        marked = True
    else:
        for markedFileName in listMarked:
            if os.path.basename(markedFileName).startswith(md5sumFirst10):
                print("File {0} already marked!".format(filePath))
                marked = True
#            im = plt.imread(markedfilepath)
#            imgshape = im.shape
#            extent = (0, imgshape[1], 0, imgshape[0])
#            implot = plt.imshow(im, extent=extent)
#            plt.show()
    if not marked:
        print filePath
        xPos = None
        yPos = None
        fileName = os.path.basename(filePath)
        fileBase, suffix = fileName.split(".")
        im = scipy.misc.imread(filePath, flatten=True)
        imgshape = im.shape
        extent = (0, imgshape[1], 0, imgshape[0])
        implot = plt.imshow(im, extent=extent)
        cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)
        print "Click on loop or close window if empty loop"
        plt.title(fileBase)
#        newFileName = os.path.join(os.path.dirname(filePath), fileBase + "_marked.png")
#        print "Saving image to " + newFileName
#        plt.savefig(newFileName)
        plt.show()
        print ImagePosition.xPos, ImagePosition.yPos
        if ImagePosition.xPos is None:
            print "Empty loop!"
        else:
            while ImagePosition.xPos is not None:
                implot = plt.imshow(im, extent=extent)
                cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)
                plt.plot([ImagePosition.xPos], [ImagePosition.yPos], marker='+', markeredgewidth=2, markersize=20, color='red')
                plt.axis(extent)
                xPos = ImagePosition.xPos
                yPos = ImagePosition.yPos
                ImagePosition.xPos = None
                ImagePosition.yPos = None
                plt.title(fileBase)
                print "Click to reposition marker or close window if marker correctly set"
                plt.show()
        if xPos is None:
            newFileName = "{0}_None_None.{1}".format(md5sumFirst10, suffix)
        else:
            newFileName = "{0}_{1:04d}_{2:04d}.{3}".format(md5sumFirst10,
                                                           int(round(float(xPos), 0)),
                                                           int(round(float(yPos), 0)),
                                                           suffix)
        print(newFileName)
        shutil.copy(filePath, os.path.join(savePath, newFileName))
