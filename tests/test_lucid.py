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
Lucid 3 project - test code
"""

__author__ = "Olof Svensson"
__contact__ = "svensson@esrf.eu"
__copyright__ = "ESRF, 2017"
__updated__ = "2018-08-20"

import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import os
import cv2
import math
import time
import glob
import numpy
import lucid3
import shutil
import unittest
import scipy.misc
from scipy import ndimage

class ImagePosition(object):
    xPos = None
    yPos = None

def onclick(event):
    if event.xdata != None and event.ydata != None:
        print(event.xdata, event.ydata)
        ImagePosition.xPos = event.xdata
        ImagePosition.yPos = event.ydata



class Test(unittest.TestCase):


    def test_lucid3(self):
        print("OpenCV Version : %s " % cv2.__version__)
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id30a1/snapshots/snapshots_20160718-152813_Gow8z5"
#        path = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id30a1/snapshots/*/*_???.png"
        # path = "/scisoft/pxsoft/data/lucid/reference/id30a1/*.png"
        # path = "/scisoft/pxsoft/data/lucid/reference/id23eh1/*.png"
        # path = "/scisoft/pxsoft/data/lucid/reference/id23eh2/*.png"
        # path = "/scisoft/pxsoft/data/lucid/reference/id30b/*.png"
        path = "/scisoft/pxsoft/data/lucid/reference/*/*.png"
        # path = "/tmp_14_days/svensson/lucid3/failed/20180829-155630/*.png"
        dateTime = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
        failedPath = os.path.join("/tmp_14_days/svensson/", "lucid3", "failed", dateTime)
        failedPathMarked = os.path.join("/tmp_14_days/svensson/", "lucid3", "failed_marked", dateTime)
        successPath = os.path.join("/tmp_14_days/svensson/", "lucid3", "success")
        if not os.path.exists(failedPath):
            os.makedirs(failedPath, 0o755)
        if not os.path.exists(failedPathMarked):
            os.makedirs(failedPathMarked, 0o755)
        if not os.path.exists(successPath):
            os.makedirs(successPath, 0o755)
        rotation = None
        maxDiff = None
        index = 0
        for filePath in glob.glob(path):
            index += 1
            print("*"*80)
            print("Image no {0}".format(index))
            print(filePath)
            if "id23eh2" in filePath:
                rotation = -90.0
            else:
                rotation = None
            fileName = os.path.basename(filePath)
            fileTitle, suffix = fileName.split(".")
            listStr = fileTitle.split("_")
            md5sum, xPosRef, yPosRef = listStr[0:3]
            if len(listStr) == 4 and listStr[3] == "vertical":
                rotation = -90.0
            image = scipy.misc.imread(filePath, flatten=True)
            imgshape = image.shape
            print(imgshape)
            maxDiff = math.sqrt(imgshape[0] ** 2 + imgshape[1] ** 2) / 15.0
            print(maxDiff)
            extent = (0, imgshape[1], 0, imgshape[0])
            result = lucid3.find_loop(filePath, rotation=rotation, debug=False)  # , rotation=rotation)
            print(result)
            resultOk = False
            xPos = None
            yPos = None
            if result[0] == 'Coord':
                xPos = result[1]
                yPos = imgshape[0] - result[2]
                if xPosRef != "None":
                    diffX = math.fabs(xPos - float(xPosRef))
                    diffY = math.fabs(yPos - float(yPosRef))
                    if max(diffX, diffY) <= maxDiff:
                        resultOk = True
            elif xPosRef == "None":
                resultOk = True
            if resultOk:
                pass
#                implot = plt.imshow(image, extent=extent)
#                plt.title(fileName)
#                if xPosRef != "None":
#                    plt.plot(float(xPosRef), float(yPosRef), marker='+', markeredgewidth=2,
#                             markersize=25, color='black')
#                if xPos is not None:
#                    plt.plot(xPos, yPos, marker='+', markeredgewidth=2,
#                             markersize=25, color='red')
#                newFileName = os.path.join(successPath, fileTitle + "_marked." + suffix)
#                print "Saving image to " + newFileName
#                plt.savefig(newFileName)
#                plt.close()
            else:
                # result = lucid3.find_loop(filePath, rotation=rotation, debug=True)
                shutil.copy(filePath, failedPath)
                ImagePosition.xPos = None
                ImagePosition.yPos = None
                # result = lucid3.find_loop(filePath, debug=True)  # , rotation=rotation)
                implot = plt.imshow(image, extent=extent)
                plt.title(fileName)
                if xPosRef != "None":
                    plt.plot(float(xPosRef), float(yPosRef), marker='+', markeredgewidth=2,
                             markersize=20, color='black')
                if xPos is not None:
                    plt.plot(xPos, yPos, marker='+', markeredgewidth=2,
                             markersize=20, color='red')
                cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)
                newFileName = os.path.join(failedPathMarked, fileTitle + "_marked." + suffix)
                print("Saving image to " + newFileName)
                plt.savefig(newFileName)
                plt.show()
                plt.close()
                if False:
                    if ImagePosition.xPos is not None:
                        implot = plt.imshow(image, extent=extent)
                        plt.title(fileName)
                        if xPosRef != "None":
                            plt.plot(float(xPosRef), float(yPosRef), marker='+', markeredgewidth=2,
                                     markersize=20, color='black')
                        if xPos is not None:
                            plt.plot(xPos, yPos, marker='+', markeredgewidth=2,
                                     markersize=20, color='red')
                        plt.plot(float(ImagePosition.xPos), float(ImagePosition.yPos), marker='+', markeredgewidth=2,
                                 markersize=20, color='green')
                        plt.show()
                        plt.close()
                        if rotation is None:
                            newFileName = "{0}_{1:04d}_{2:04d}.{3}".format(md5sum,
                                                                           int(round(float(ImagePosition.xPos), 0)),
                                                                           int(round(float(ImagePosition.yPos), 0)),
                                                                           suffix)
                        else:
                            newFileName = "{0}_{1:04d}_{2:04d}_vertical.{3}".format(md5sum,
                                                                           int(round(float(ImagePosition.xPos), 0)),
                                                                           int(round(float(ImagePosition.yPos), 0)),
                                                                           suffix)
                    elif rotation is None:
                        newFileName = "{0}_None_None.{1}".format(md5sum, suffix)
                    else:
                        newFileName = "{0}_None_None_vertical.{1}".format(md5sum, suffix)
                    print("Old file name: {0}".format(fileName))
                    print("New file name: {0}".format(newFileName))
                    saveChanges = raw_input("Save changes? yes/no: ")
                    if saveChanges.lower() == "yes":
                        os.rename(filePath, os.path.join(os.path.dirname(filePath), newFileName))
                        print("Changes saved.")



            # plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
