'''
Created on Jul 19, 2016

@author: svensson
'''

import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import os
import cv2
import math
import glob
import numpy
import lucid3
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
        # path = "/scisoft/pxsoft/data/lucid/reference/id23eh2/*.png"
        # path = "/scisoft/pxsoft/data/lucid/reference/id30b/*.png"
        path = "/scisoft/pxsoft/data/lucid/reference/*/*.png"
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id29/snapshots/20170823"
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id23eh2/snapshots/2070704"
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id23eh2/snapshots/20170822"
        failedPath = os.path.join("/tmp_14_days/svensson/", "lucid3", "failed")
        successPath = os.path.join("/tmp_14_days/svensson/", "lucid3", "success")
        if not os.path.exists(failedPath):
            os.makedirs(failedPath, 0755)
        if not os.path.exists(successPath):
            os.makedirs(successPath, 0755)
        rotation = None
        maxDiff = None
        for filePath in glob.glob(path):
            if "id23eh2" in filePath:
                rotation = -90.0
            else:
                rotation = None
            fileName = os.path.basename(filePath)
            fileTitle, suffix = fileName.split(".")
            md5sum, xPosRef, yPosRef = fileTitle.split("_")
            # print(md5sum, xPosRef, yPosRef)
            # print(filePath)
            image = scipy.misc.imread(filePath, flatten=True)
            # image = scipy.misc.imrotate(image, -90)
            # im = plt.imread(filePath)
            # im = ndimage.rotate(im, -90)
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
                implot = plt.imshow(image, extent=extent)
                plt.title(fileName)
                if xPosRef != "None":
                    plt.plot(float(xPosRef), float(yPosRef), marker='+', markeredgewidth=2,
                             markersize=25, color='black')
                if xPos is not None:
                    plt.plot(xPos, yPos, marker='+', markeredgewidth=2,
                             markersize=25, color='red')
                newFileName = os.path.join(successPath, fileTitle + "_marked." + suffix)
                print "Saving image to " + newFileName
                plt.savefig(newFileName)
                plt.close()
            else:
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
                newFileName = os.path.join(failedPath, fileTitle + "_marked." + suffix)
                print "Saving image to " + newFileName
                plt.savefig(newFileName)
                plt.show()
                plt.close()
                if ImagePosition.xPos is not None:
                    implot = plt.imshow(image, extent=extent)
                    plt.title(fileName)
                    if xPos is not None:
                        plt.plot(xPos, yPos, marker='+', markeredgewidth=2,
                                 markersize=20, color='black')
                    if xPosRef != "None":
                        plt.plot(float(xPosRef), float(yPosRef), marker='+', markeredgewidth=2,
                                 markersize=20, color='red')
                    plt.plot(float(ImagePosition.xPos), float(ImagePosition.yPos), marker='+', markeredgewidth=2,
                             markersize=20, color='green')
                    plt.show()
                    plt.close()
                    newFileName = "{0}_{1:04d}_{2:04d}.{3}".format(md5sum,
                                                                   int(round(float(ImagePosition.xPos), 0)),
                                                                   int(round(float(ImagePosition.yPos), 0)),
                                                                   suffix)
                else:
                    newFileName = "{0}_None_None.{1}".format(md5sum, suffix)
                print(newFileName)
                saveChanges = raw_input("Save changes? yes/no: ")
                if saveChanges.lower() == "yes":
                    os.rename(filePath, os.path.join(os.path.dirname(filePath), newFileName))



            # plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
