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
__copyright__ = "ESRF, 2017"
__updated__ = "2022-02-01"

# This code is a complete re-factoring of the "lucid2" code written by Sogeti,
# see : https://github.com/mxcube/lucid2

import os
import cv2
import shutil
import tempfile
import imageio
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa E402

# Find out if we are using OpenCV version 3:
if cv2.__version__.split(".")[0] == "3":
    OPENCV3 = True
else:
    OPENCV3 = False

# Offset applied to shrink initial image in order to avoid border effect
SHRINK_OFFSET = (4, 4)

# White border applied to extend image in order to avoid border effect
WHITE_BORDER = (10, 10)

# Minimum relative loop area
MIN_REL_LOOP_AREA = 0.02

# Minimal lenght of detected contour (used when contour are still opened)
MIN_CONTOUR_LENGTH = 125


AREA_POINT_REL = 0.005

# This parameter indicate the number of iteration on closing algorithm
# (upper value could lead to dust agglomeration with support)
CLOSING_ITERATIONS = 6

# Possible Criteron modes
CRIT_MOD_NOVALUE = 0
CRIT_MOD_SUP = 1
CRIT_MOD_LOOP = 2
CRIT_MOD_NARROW = 3

# Default threshold value
DEFAULT_THRESHOLD = 20

# Default enhanced contrast threshold value
ENHANCED_CONTRAST_REL_THRESHOLD = 0.25
ENHANCED_CONTRAST_REL_RADIUS = 0.8
ENHANCED_CONTRAST_MAX_THRESHOLD = 25


def find_loop(
    image,
    rotation=False,
    debug=False,
    archiveDir=None,
    IterationClosing=CLOSING_ITERATIONS,
):
    """
    This function detect support (or loop).
    in : filename : string image Filename / Format accepted :
    Out : tuple of coordinates : (string status, float x, float y) where string 'status' take value 'Coord'
                                 or 'No loop detected' depending if loop was detected or not.
                                 If no loop was detected X and y take the value -1.
    """
    # Archive the image if archiveDir is not None
    # if archiveDir is not None and type(image) == str:
    #     archiveImage(filename, archiveDir)

    # Load the image as a numpy array, taking into account rotation

    if type(image) == str:
        grayImage = loadGrayImage(image, rotation)
    elif type(image) == np.ndarray:
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
        grayImage = R * 299.0 / 1000 + G * 587.0 / 1000 + B * 114.0 / 1000
    else:
        print("ERROR : Input image could not be opened, check format or path")
        return (
            "ERROR : Input image could not be opened, check format or path",
            -10,
            -10,
        )

    debugPlot(grayImage, "Initial gray image", debug)

    # Calculate min loop area
    rows, cols = grayImage.shape
    imageArea = rows * cols
    minLoopArea = imageArea * MIN_REL_LOOP_AREA
    if debug:
        print("Min loop area: {0}".format(minLoopArea))

    # Image analysis

    # Step 1 :
    # Removing borders of size SHRINK_OFFSET from image
    grayImageTruncated = grayImage[
        SHRINK_OFFSET[0] : rows - 2 * SHRINK_OFFSET[0],
        SHRINK_OFFSET[1] : cols - 2 * SHRINK_OFFSET[1],
    ]
    debugPlot(grayImageTruncated, "Gray image truncated", debug)

    # Step 2 :
    # Smoothen the image with gaussian blur
    blurredImage = cv2.GaussianBlur(grayImageTruncated, ksize=(11, 9), sigmaX=0)
    debugPlot(blurredImage, "Gray image blurred", debug)

    # Step 3 :
    # Enhance contrast for very smooth images within a centered circle
    enhancedContrastImage = enhanceContrast(blurredImage)
    debugPlot(enhancedContrastImage, "Enhanced contrast image", debug)

    # Step 4 :
    # Apply Laplacian operator on image
    laplacianImage = cv2.Laplacian(enhancedContrastImage, cv2.CV_64F, ksize=5)
    debugPlot(laplacianImage, "Laplacian image", debug)

    # Step 5 :
    # Applying again SHRINK_OFFSET to avoid border effect
    laplacianImageTruncated = laplacianImage[
        SHRINK_OFFSET[0] : rows - 2 * SHRINK_OFFSET[0],
        SHRINK_OFFSET[1] : cols - 2 * SHRINK_OFFSET[1],
    ]
    debugPlot(laplacianImageTruncated, "Laplacian image truncated", debug)

    # Step 6 :
    # Apply an asymetrique  smoothing
    # if rotation is None:
    smoothLaplacianImage = cv2.GaussianBlur(
        laplacianImageTruncated, ksize=(21, 11), sigmaX=0
    )
    debugPlot(smoothLaplacianImage, "Smooth laplacian image truncated", debug)

    # Step 7 :
    # Morphology examination
    MKernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(5, 3), anchor=(3, 1)
    )
    morphologyExImage = cv2.morphologyEx(
        smoothLaplacianImage, cv2.MORPH_CLOSE, MKernel, iterations=IterationClosing
    )
    morphologyExImage[np.where(morphologyExImage < 0)] = 0
    morphologyExImage[np.where(morphologyExImage >= 255)] = 0
    morphologyExImage = np.uint8(morphologyExImage)
    debugPlot(morphologyExImage, "Morphology image", debug)

    # Step 8 :
    # Add white border to image
    whiteBorderImage = addWhiteBorder(
        morphologyExImage[:], WHITE_BORDER[0], WHITE_BORDER[1]
    )
    debugPlot(whiteBorderImage, "White border image", debug)

    # Step 9 : Compute and apply threshold
    threshold = computeThreshold(whiteBorderImage)
    if debug:
        print("Threshold: {0}".format(threshold))
    ret, thresholdImage = cv2.threshold(whiteBorderImage, threshold, 255, 0)
    debugPlot(thresholdImage, "Threshold image", debug)

    # Step 9a : Right white border
    rightBorder = addWhiteBorderRight(
        thresholdImage[:], WHITE_BORDER[0], WHITE_BORDER[1]
    )
    debugPlot(thresholdImage, "Right border image", debug)

    # Step 10: Gaussian smoothing
    imageSmoothThreshold = cv2.GaussianBlur(rightBorder, ksize=(11, 11), sigmaX=0)
    debugPlot(imageSmoothThreshold, "smooth threshold image", debug)

    # Step 11 : Compute and apply threshold
    # Convert grayscale laplacian image to binarie image using "threshold" as threshold value
    ret, imageSmoothThreshold2 = cv2.threshold(
        imageSmoothThreshold, threshold, 255, cv2.THRESH_BINARY_INV
    )
    debugPlot(imageSmoothThreshold2, "Second smooth threshold image", debug)

    # Step 12 : Edge detection
    imageEdges = cv2.Canny(imageSmoothThreshold2, 0, 2)
    debugPlot(imageEdges, "Edge detection", debug)

    # Step 13 : Dilate edges in order to close contours
    kernel = np.ones((3, 3), np.uint8)
    imageEdgesDilated = cv2.dilate(imageEdges, kernel, iterations=1)
    debugPlot(imageEdgesDilated, "Edges dilated", debug)

    # Step 14 : Find contours
    if OPENCV3:
        img, contours, hierarchy = cv2.findContours(
            imageEdgesDilated, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        contours, hierarchy = cv2.findContours(
            imageEdgesDilated, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )

    maxAreaContour = findMaxAreaContour(contours, minLoopArea, debug=debug)
    if debug:
        print("Max area contour: {0}".format(maxAreaContour))

    if maxAreaContour is None:
        # No loop is detected
        point = ("No loop detected", -1, -1)
    else:
        rows, cols = imageSmoothThreshold2.shape

        #
        indices = mapContour(maxAreaContour, cols)

        #     The coordinate of target is computed in the traited image
        pointShifted = findLoopPoint(indices, maxAreaContour)

        #     The coordinate in original image are computed taken into account SHRINK_OFFSET and white bordure
        point = (
            pointShifted[0],
            pointShifted[1] + 2 * SHRINK_OFFSET[0] - WHITE_BORDER[0],
            pointShifted[2] + 2 * SHRINK_OFFSET[1] - WHITE_BORDER[1],
        )

        # Mask the lower and upper right corners
        if point[1] < cols * 0.2 and (
            point[2] < rows * 0.2 or (rows - point[2]) < rows * 0.2
        ):
            # No loop is detected
            point = ("No loop detected", -1, -1)
        else:
            if rotation:
                point = (point[0], point[2], cols - point[1])
            if debug:
                if type(image) == str:
                    fileBase = os.path.splitext(os.path.basename(image))[0]
                    image = imageio.imread(image, as_gray=True)
                else:
                    fileBase = "Image"
                imgshape = image.shape
                extent = (0, imgshape[1], 0, imgshape[0])
                implot = plt.imshow(image, extent=extent, cmap="gray")
                plt.title(os.path.basename(fileBase))
                if point[0] == "Coord":
                    xPos = point[1]
                    yPos = imgshape[0] - point[2]
                    plt.plot(
                        xPos,
                        yPos,
                        marker="+",
                        markeredgewidth=2,
                        markersize=20,
                        color="red",
                    )
                # newFileName = os.path.join(archiveDir, fileBase + "_marked.png")
                # print("Saving image to " + newFileName)
                # plt.savefig(newFileName)
                plt.show()
                plt.close()

    return point


def archiveImage(filename, archiveDir):
    (file_descriptor, fileBase) = tempfile.mkstemp(prefix="lucid2_", dir=archiveDir)
    os.close(file_descriptor)
    suffix = os.path.splitext(filename)[1]
    shutil.copy(filename, fileBase + suffix)
    os.remove(os.path.join(archiveDir, fileBase))


def debugPlot(image, title, debug=False):
    if debug:
        plt.title(title)
        plt.imshow(image)
        plt.show()


def loadGrayImage(filename, rotation=False):
    # Read image
    image = cv2.imread(filename)
    # Convert to image
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take into account rotation
    if rotation:
        # Image assumed to be rotated 90 degrees anti-clockwise
        grayImage = np.rot90(grayImage, k=3)
    return grayImage


def enhanceContrast(image):
    enhancedContrastImage = image.copy()
    maxImage = np.max(enhancedContrastImage)
    # print(maxImage)
    rows, cols = image.shape
    center = [int(cols / 2), int(rows / 2)]
    # print(center)
    radius = min(center[0], center[1]) * ENHANCED_CONTRAST_REL_RADIUS
    # print(radius)
    yGrid, xGrid = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((xGrid - center[0]) ** 2 + (yGrid - center[1]) ** 2)
    mask1 = dist_from_center <= radius
    mask2 = np.ones((rows, cols), dtype=image.dtype) - mask1
    img_gray_masked = image * mask1 + mask2 * 100
    # debugPlot(img_gray_masked, "Mask 1", True)
    contrastThreshold = min(
        ENHANCED_CONTRAST_REL_THRESHOLD * maxImage, ENHANCED_CONTRAST_MAX_THRESHOLD
    )
    # print(contrastThreshold)
    enhancedContrastImage[np.where(img_gray_masked < contrastThreshold)] = 0
    return enhancedContrastImage


def findMaxAreaContour(contours, minLoopArea, debug=False):
    """
    This function finds the contour with area > minLoopArea or length > MIN_CONTOUR_LENGTH
    """
    maxAreaContour = None
    maxContour = None
    maxLength = None
    for contour in contours:
        length = len(contour)
        area = cv2.contourArea(contour)
        # print(area, length)
        if area > minLoopArea or length > MIN_CONTOUR_LENGTH:
            if maxAreaContour is None or maxAreaContour < area:
                maxAreaContour = area
                maxContour = contour
                maxLength = length
    if debug:
        print("Contour area: {0}".format(maxAreaContour))
        print("Contour length: {0}".format(maxLength))
    return maxContour


def addWhiteBorder(img, XSize, YSize):
    """
    This fonction add white border to an image

    In : img : numpy array : input image
    In : XSize : int: width of white border
    In : YSize : int: heigh of white border
    Out : ouput_mat : numpy array : Output image copy of input one added of white border
    """
    s0, s1 = img.shape
    dtypeI = img.dtype
    output_mat = np.zeros((s0, s1), dtype=dtypeI)
    output_mat[XSize : s0 - XSize, YSize:s1] = img[XSize : s0 - XSize, 0 : s1 - YSize]
    # Clear corners
    cutoff_size = 50
    output_mat[0:cutoff_size, 0:cutoff_size] = 0
    output_mat[s0 - cutoff_size : s0, 0:cutoff_size] = 0
    return output_mat


def addWhiteBorderRight(img, XSize, YSize):
    """
    This fonction add white border to an image

    In : img : numpy array : input image
    In : XSize : int: width of white border
    In : YSize : int: heigh of white border
    Out : ouput_mat : numpy array : Output image copy of input one added of white border
    """
    s0, s1 = img.shape
    dtypeI = img.dtype
    output_mat = np.zeros((s0, s1 + YSize), dtype=dtypeI)
    output_mat[:, 0:s1] = img[:, :]
    # Clear corners
    cutoff_size = 50
    output_mat[0:cutoff_size, 0:cutoff_size] = 0
    output_mat[s0 - cutoff_size : s0, 0:cutoff_size] = 0
    return output_mat


def computeThreshold(image):
    """
    This fonction compute threshold value. In first the image's histogram is calculated.
    The threshold value is set to the first indexe of histogram wich respect the
    following criterion : DH > 0, DH(i)/H(i) > 0.1 , H(i) < 0.01 % of the Norm.

    In : image : ipl Image : image to treated
    Out: threshold : Int : Value of the threshold
    """
    dim = 255
    norm = image.shape[0] * image.shape[1]
    bins = [float(x) for x in range(dim)]
    hist, bin_edges = np.histogram(image, bins)
    norm = norm - hist[0]
    i = 1
    som = 0
    while som < 0.8 * norm and i < len(hist) - 1:
        som = som + hist[i]
        i = i + 1
    while (
        ((hist[i] - hist[i - 1]) < 0)
        or (int((hist[i] - hist[i - 1]) / hist[i - 1]) > 0.1)
        or (hist[i] > 0.01 * norm)
    ) and (i < len(hist) - 1):
        i = i + 1
        if hist[i - 1] == 0:
            break

    if i == len(hist) - 1:
        threshold = 0
    else:
        threshold = i

    # Default threshold value if no threshold found
    if threshold == 0:
        threshold = DEFAULT_THRESHOLD

    # print("Seuil: {0}".format(threshold))

    return threshold


def mapContour(contour, s0):
    """
    This function transform a list of coordinate X(i) into a function of coordinate i(X)

    In : contour : Contour represented by a sequence of point
    In : s0 :
    Out : ListInd : List : function indexe(abscissa)

    """
    listIndices = []
    for i in range(0, s0):
        listResult = list(np.where(contour[:, :, 0] == i)[0])
        listIndices.append(listResult)
    return listIndices


def findPointMax(listInd):
    """
    This function return the maximal not null value in a list

    In : list : List of index
    Out : maximal indexe
    """
    i = len(listInd) - 1
    while (listInd[i] == [] or listInd[i] is None) and i >= 0:
        i = i - 1
    if i == 0:
        return None
    else:
        return listInd[i][0]


def checkIfX(coordinate, x):
    """
    This function is a filter which return true if the first value of tuple is equal to x
    In : tuple : tuple to be tested
    In : x : float, The test value
    Out : Bool : Result of test
    """
    if coordinate[0][0] == x:
        return True
    else:
        return False


def findLoopPoint(listInd, contour):
    """
    This function integrates contour, in order to extract target coordinates
    In : listInd : list : list of indices contour i(X) where x is the abscissa of contour point
    In : seq : list of int tuple (X,Y) : list of point of contour
    Out : tuple : Coordinate of the target
    """
    # Initialize both cover indexe to the one of the maximal abscissa point of contour
    indMax = findPointMax(listInd)
    # iteration number initialisation
    xcib = None
    ycib = None
    noIterations = 0
    search = True
    # if sequence is not empty
    if indMax is not None:
        # Get the lengh of the sequence
        s0 = len(contour)
        # Initialize both cover indexe to the one of the maximal abscissa point of contour
        indp = indMax
        indm = indMax
        # Initialize refrence distance to zero
        distRef = 0
        # Initialize the maximal value of abscissa
        xtot = contour[indMax][0][0]
        # Get the criter for extract coordinate point
        pointDetectionCriteria, contourArea = getPointDetectionCriteria(
            listInd, contour, indMax
        )
        areaPointRel = contourArea * AREA_POINT_REL
        Nmax = s0 / 2.0
        # While coordinates point are not found and iteration max is not reached
        while search and noIterations < Nmax:
            noIterations = noIterations + 1
            # Only one is decrease between upper index and lower index, it's the one with the lower value. Bounding condition are applied on indexes
            if contour[indp][0][0] > contour[indm][0][0]:
                if indp < s0 - 2:
                    indp = indp + 1
                else:
                    indp = 0
            else:
                if indm > 1:
                    indm = indm - 1
                else:
                    indm = s0 - 1
            poids = float(abs(contour[indp][0][0] - contour[indm][0][0])) / float(xtot)
            dist = abs(contour[indp][0][1] - contour[indm][0][1])
            distRef = distRef + dist * poids
            # Partial Area of contour is calculated
            if indm < indp:
                AreaTmp = cv2.contourArea(contour[indm:indp])
            else:
                AreaTmp = cv2.contourArea(contour) - cv2.contourArea(contour[indp:indm])
            AreaTmp = abs(AreaTmp)
            # if Partial area is lower than area criteron and the criteron mod is not Narrow or support
            if (
                AreaTmp < areaPointRel
                and pointDetectionCriteria != CRIT_MOD_NARROW
                and pointDetectionCriteria != CRIT_MOD_SUP
            ):
                # Coordinates are saved
                xcib = (contour[indm][0][0] + contour[indp][0][0]) * 0.5
                ycib = (contour[indm][0][1] + contour[indp][0][1]) * 0.5
            # else if criteron mod is narrow or support and distance between current point and maximal abscissa is lower than 80 microns
            elif (
                pointDetectionCriteria == CRIT_MOD_NARROW
                or pointDetectionCriteria == CRIT_MOD_SUP
            ) and (xtot - (contour[indm][0][0] + contour[indp][0][0]) * 0.5) < 40:
                # Coordinates are saved
                xcib = (contour[indm][0][0] + contour[indp][0][0]) * 0.5
                ycib = (contour[indm][0][1] + contour[indp][0][1]) * 0.5
            # else if coordinate point already found and criteron mod is not narrow nor support
            elif (
                AreaTmp > areaPointRel
                and pointDetectionCriteria != CRIT_MOD_NARROW
                and pointDetectionCriteria != CRIT_MOD_SUP
            ):
                # the loop is ended in order to avoid minimal contous abscissa interference
                search = False
            if xcib is not None and ycib is not None:
                xcib = int(xcib)
                ycib = int(ycib)
    if xcib is None or ycib is None:
        return ("No loop detected", -1, -1)
    else:
        return ("Coord", xcib, ycib)


def getPointDetectionCriteria(listInd, contour, indMax):
    """
    This function use contour to determine the type of support and the type of criteria
    to use for get point coord. The determination is based on the shape of counter,
    especially the width of counter versus abscissa.

    There are 4 different types:
    - Narrow, which is for Narrow support.
    - SUP which for support.
    - Loop for loop, all loop are not detected in this categorie, only one wich have a principal support.
    - And default value which is for all not detected support.

    In[1] List of indice depenting of value of X
    In[2] CvSeq of main contour
    In[3] Indice of the point of CvSeq having the Max X
    Out [1] : Area of interrest wich will be used as reference Area

    Area critere and Detected Type are global variable
    """

    # Definition of value used for criterion
    CRITERON_DY_LOOP_SUP = 140
    CRITERON_DEFAULT3 = 55
    CRITERON_DX_LINEARITY = 70
    CRITERON_DY_LINEARITY = 45
    CRITERON_DY_NARROW = 25
    CRITERON_DX_NARROW = 25
    CRITERON_DY_LOOP_SUP2 = 75
    #  Global variable declaration
    pointDetectionCriteria = CRIT_MOD_NOVALUE
    global AREA_POINT_REL
    AREA_POINT_REL = 0.008
    #  Local variable declaration
    pointDetectionCriteria_opt = CRIT_MOD_NOVALUE  # Used when a possible support is detected, but loop could still be detected also.
    # If there is a Maximum in X
    if indMax is not None:
        # Get the length of the contour
        s0 = len(contour)
        indp = indMax  # Initialisation of index
        indm = indMax  # Initialisation of index
        xtot = contour[indMax][0][0]  # Set Maximal value
        deltaY = [0]  # List of width of contour versus X
        listXMax = [xtot]  # List of X linked to previous list of deltaY
        yMax = 0
        xMem = 12000
        xMin = 0
        xMinMem = xtot
        xm = 600
        xMax = xtot
        noIterations = 0
        # Indexes of contours are cover on 600 micron or until abscissa 15 is reach or maximal iteration or a final criteron is foun
        while (
            (xtot - xMax) < 300
            and xm > 15
            and noIterations < 500
            and pointDetectionCriteria == CRIT_MOD_NOVALUE
        ):
            noIterations = noIterations + 1
            x1 = contour[indp][0][0]  # Upper Abscissa
            y1 = contour[indp][0][1]  # Upper Ordinate
            x2 = contour[indm][0][0]  # Lower Abscissa
            y2 = contour[indm][0][1]  # Lower Ordinate
            xm = (x1 + x2) * 0.5  # Mean Abscissa
            yd = abs(y2 - y1)  # Ordinate difference
            xMax = max(x1, x2)  # Abscissa Maximal
            xMin = min(x1, x2)  # Abscissa Minimal
            if yd > yMax:
                yMax = yd
            # If the minimal Abscissa strongly increase during one iteration, the shape should be lineare
            # Witch is a caracteristic of a kind of support
            if (
                abs(xMin - xMinMem) > CRITERON_DX_LINEARITY
                and pointDetectionCriteria_opt != CRIT_MOD_SUP
            ):
                # If the step in Y corresponding is upper than 90 Micron
                if (yMax - yd) > CRITERON_DY_LINEARITY:
                    pointDetectionCriteria = CRIT_MOD_LOOP
                    AREA_POINT_REL = 0.4
                else:
                    # An option is took to Support type, but not definitly cause some loop can present this kind of shape too
                    pointDetectionCriteria_opt = CRIT_MOD_SUP
            # If yd is upside the narrow limit and dx is downside the x narrow limit then it s not a narrow type
            if yd > CRITERON_DY_NARROW and (xtot - xMax) < CRITERON_DX_NARROW:
                AREA_POINT_REL = 0.05
            # If the CRITERON_DX_NARROW has been cover and area_point_rel is still to the default value and no support option has been detected
            if (
                (xtot - xMax) > CRITERON_DX_NARROW
                and AREA_POINT_REL < 0.04
                and pointDetectionCriteria_opt != CRIT_MOD_SUP
            ):
                # Then criteron is set to narrow mod
                pointDetectionCriteria = CRIT_MOD_NARROW
            # If a step in Dy superior to 140 micron is detected then default value for relative area is set to 0.15
            if (
                yd > CRITERON_DEFAULT3
                and AREA_POINT_REL < 0.1
                and pointDetectionCriteria_opt != CRIT_MOD_SUP
            ):
                AREA_POINT_REL = 0.15
            # If a loop support is detected for the fist time
            if yd > CRITERON_DY_LOOP_SUP and pointDetectionCriteria != CRIT_MOD_LOOP:
                Xint = 0
                iint = 0
                # Search back the Dy value 50 micron back
                while Xint < CRITERON_DX_NARROW and iint < len(deltaY):
                    Xint = listXMax[iint] - xMax
                    iint = iint + 1
                DY25 = yd - deltaY[iint - 1]
                # if the step is taller than 140 micron then mod is loop and relative area is set to 0.2
                if DY25 > CRITERON_DY_LOOP_SUP2:
                    pointDetectionCriteria = CRIT_MOD_LOOP
                    AREA_POINT_REL = 0.2
                # Else criteron mode is set to support
                else:
                    pointDetectionCriteria = CRIT_MOD_SUP
            # The calculations made on previous indexe ar keep in memory if contour do not present irregularity
            # if abscissa is decreasing
            if (xMax - xMem) < 0:
                listXMax.insert(0, xMax)
                xMem = xMax
                deltaY.insert(0, yd)
            # else the yd value is test and keep if it's highter than saved one
            else:
                i = 0
                if yd > deltaY[0]:
                    while listXMax[0] < xMax:
                        i = i + 1
                        listXMax.pop(0)
                        deltaY.pop(0)
            # Only one is decrease between upper index and lower index, it's the one with the lower value. Bounding condition are applied on indexes
            if contour[indp][0][0] > contour[indm][0][0]:
                if indp < s0 - 2:
                    indp = indp + 1
                else:
                    indp = 0
            else:
                if indm > 1:
                    indm = indm - 1
                else:
                    indm = s0 - 1
    # Depending on criteron mode the reference area is computed
    if pointDetectionCriteria == CRIT_MOD_LOOP:
        if indm < indp:
            contourArea = cv2.contourArea(contour[indm:indp])
        else:
            contourArea = cv2.contourArea(contour) - cv2.contourArea(contour[indp:indm])
    elif pointDetectionCriteria_opt == CRIT_MOD_SUP:
        pointDetectionCriteria = CRIT_MOD_SUP
        contourArea = cv2.contourArea(contour)
    else:
        contourArea = cv2.contourArea(contour)

    return pointDetectionCriteria, contourArea
