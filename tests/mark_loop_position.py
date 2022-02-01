"""
Created on Oct 15, 2014

@author: svensson
"""

import matplotlib.image
import matplotlib.pyplot as plt
import glob, os
import scipy.misc
import pylab
import numpy
import tempfile


class ImagePosition(object):
    xPos = None
    yPos = None


def onclick(event):
    if event.xdata != None and event.ydata != None:
        print (event.xdata, event.ydata)
        ImagePosition.xPos = event.xdata
        ImagePosition.yPos = event.ydata


snapshotDir = "/scisoft/pxsoft/data/WORKFLOW_TEST_DATA/id30a1/snapshots/snapshots_20160718-152813_Gow8z5"
savePath = os.path.join("/tmp_14_days/svensson/", "test2")
if not os.path.exists(savePath):
    os.makedirs(savePath, 0755)
(coordfd, coordfilename) = tempfile.mkstemp(suffix=".txt", prefix="coord", dir=savePath)
coordfile = open(coordfilename, "w")
for filePath in glob.glob(os.path.join(snapshotDir, "*.png")):
    ImagePosition.xPos = None
    ImagePosition.yPos = None
    # Check if marked image exists
    if not filePath.split(".")[0].endswith("marked"):
        markedfilepath = filePath.split(".")[0] + "_marked.png"
        print markedfilepath
        if os.path.exists(markedfilepath):
            im = plt.imread(markedfilepath)
            imgshape = im.shape
            extent = (0, imgshape[1], 0, imgshape[0])
            implot = plt.imshow(im, extent=extent)
            plt.show()
        fileName = os.path.basename(filePath)
        fileBase = fileName.split(".")[0]
        print fileBase
        im = plt.imread(filePath)
        imgshape = im.shape
        extent = (0, imgshape[1], 0, imgshape[0])
        implot = plt.imshow(im, extent=extent)
        cid = implot.figure.canvas.mpl_connect("button_press_event", onclick)
        print "Click on loop or close window if empty loop"
        plt.title(fileBase)
        #        newFileName = os.path.join(os.path.dirname(filePath), fileBase + "_marked.png")
        #        print "Saving image to " + newFileName
        #        plt.savefig(newFileName)
        plt.show()
        #    print ImagePosition.xPos, ImagePosition.yPos
        if ImagePosition.xPos is None:
            print "Empty loop!"
            coordfile.write("%20s %3d %3d\n" % (fileName, -1, -1))
        else:
            while ImagePosition.xPos is not None:
                implot = plt.imshow(im, extent=extent)
                cid = implot.figure.canvas.mpl_connect("button_press_event", onclick)
                plt.plot(
                    [ImagePosition.xPos],
                    [ImagePosition.yPos],
                    marker="+",
                    markeredgewidth=2,
                    markersize=20,
                    color="red",
                )
                coordfile.write(
                    "%30s %5d %5d\n"
                    % (fileName, int(ImagePosition.xPos), int(ImagePosition.yPos))
                )
                coordfile.flush()
                plt.axis(extent)
                ImagePosition.xPos = None
                ImagePosition.yPos = None
                plt.title(fileBase)
                # print "Saving image to " + newFileName
                # plt.savefig(newFileName)
                print "Click to reposition marker or close window if marker correctly set"
                plt.show()
