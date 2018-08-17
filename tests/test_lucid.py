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

class Test(unittest.TestCase):


    def test_lucid3(self):
        print("OpenCV Version : %s " % cv2.__version__)
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id30a1/snapshots/snapshots_20160718-152813_Gow8z5"
#        path = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id30a1/snapshots/*/*_???.png"
        path = "/scisoft/pxsoft/data/lucid/reference/id30a1/*.png"
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id29/snapshots/20170823"
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id23eh2/snapshots/2070704"
#        directory = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id23eh2/snapshots/20170822"
        savePath = os.path.join("/tmp_14_days/svensson/", "lucid3")
        if not os.path.exists(savePath):
            os.makedirs(savePath, 0755)
        rotation = None
        maxDiff = 50
        for filePath in glob.glob(path):
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
            extent = (0, imgshape[1], 0, imgshape[0])
            result = lucid3.find_loop(filePath, debug=False)  # , rotation=rotation)
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
            if not resultOk:
                # result = lucid3.find_loop(filePath, debug=True)  # , rotation=rotation)
                implot = plt.imshow(image, extent=extent)
                plt.title(fileName)
                if xPos is not None:
                    plt.plot(xPos, yPos, marker='+', markeredgewidth=2,
                             markersize=20, color='black')
                if xPosRef != "None":
                    plt.plot(float(xPosRef), float(yPosRef), marker='+', markeredgewidth=2,
                             markersize=20, color='red')
                newFileName = os.path.join(savePath, fileTitle + "_marked." + suffix)
                print "Saving image to " + newFileName
                plt.savefig(newFileName)
                plt.show()
                plt.close()
            # plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
