import numpy as np
import cv2 as cv
from time import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm

ridgeDetection = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])

identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobelY = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

kernel3 = np.array([
    [-1, -2, 0, -2, -1],
    [-2, -3, 0, -3, -2],
    [0, 0, 0, 0, 0],
    [2, 3, 0, 3, 2],
    [1, 2, 0, 2, 1]
])

gaussian = np.array([
    [1,4,6,4,1],
    [4,16,24,16,4],
    [6,24,36,24,6],
    [4,16,24,16,4],
    [1,4,6,4,1],
])
gaussian = np.multiply(gaussian, 1/256)

class superpixel():
    def __init__(self, startPixel:tuple, blur:int, checkRange:int):
        self.classification = None
        self.dtype = np.dtype([("x", '<u2'), ("y", '<u2'), ('d', '<u8')])
        self.pixelData = np.array([startPixel],dtype=self.dtype)
        self.setPixelIntensity(startPixel[2])
        self._setIntensityRange(startPixel[2], blur=blur)
        self.checkRange = checkRange

    def addpixel(self, pixel:tuple):
        newPixel = np.array([pixel],dtype=self.dtype)
        self.pixelData = np.append(self.pixelData, newPixel)

    def checkpixel(self, pixel:tuple):
        xneighbor = False
        yneighbor = False

        if np.isin(np.arange(pixel[0]-self.checkRange,pixel[0]+self.checkRange), self.pixelData["x"]).any():
            #print("Pixel has an x neighbor")
            xneighbor = True

        if np.isin(np.arange(pixel[1]-self.checkRange,pixel[1]+self.checkRange), self.pixelData["y"]).any():
            #print("Pixel has a y neighbor")
            yneighbor = True
        
        if xneighbor and yneighbor:
            if np.isin(pixel[2], self.intensityRange).any():
                return True
            else:
                return False
        else:
            return False

    def _setIntensityRange(self, startValue:int, blur:int):
        self.intensityRange = np.arange(startValue - blur,startValue + blur)

    def assignColour(self,rgbValue:np.ndarray):
        self.colour = rgbValue
        pass

    def generateSuperPixelArray(self, size:tuple):
        pixelArray = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        pixelArray[self.pixelData["x"], self.pixelData["y"], :] = self.colour
        return pixelArray

    def generateMask(self, size:tuple):
        mask = np.zeros(size, dtype= np.bool8)
        mask[self.pixelData["x"], self.pixelData["y"]] = 1
        invertedMask = np.bitwise_not(mask)
        return mask, invertedMask

    def setPixelIntensity(self, intensity:int):
        self.intensity = intensity
    
    def getPixelIntensity(self):
        return self.intensity

    def getSuperPixelData(self):
        return self.pixelData

    def setClassification(self, label:str):
        self.classification = label
    
    def getClassification(self):
        return self.classification

def conv2D(img, kernel, paddingType:str = None):

    # Initialise output array that is the same size as input array
    out = np.zeros(np.shape(img))

    # Pad input array such that the entire kernel has values to act on
    padding = np.asarray(np.divide(np.shape(kernel),2), dtype=np.int16)
    
    if paddingType:
        img = np.pad(img,padding, paddingType)
    else:
        img = np.pad(img,padding)
    
    
    for row in range(0+padding[0],np.shape(img)[0]-padding[0]):

        for column in range(0+padding[1],np.shape(img)[1]-padding[1]):
            
            startRow = row - padding[0]
            
            startCol = column - padding[1]
            
            endRow = row + padding[0] + 1
            
            endCol = column + padding[1] + 1

            out[row-padding[0],column-padding[1]] = np.sum(np.multiply(img[startRow:endRow,startCol:endCol],kernel))
    return(out)

def rescaleImage(img,scale):
    img = np.repeat(img, scale, axis=0)
    img = np.repeat(img, scale, axis=1)
    return img

def normalize(img):
    if img.min() < 0:
        img = np.add(np.abs(img.min()), img)
        if img.max() > 0:
            img = np.divide(img, img.max())
    else:
        img = np.subtract(img, np.abs(img.min()))
        if img.max() > 0:
            img = np.divide(img, img.max())
    return np.asarray(img,dtype=np.float32)

def convertTo8Bit(img):
    img = normalize(img)
    outputImg = np.asarray(np.multiply(img, 255), dtype=np.uint8)
    return outputImg

def convertTo3Channel(img):
    size = np.shape(img)

    if len(size) <= 2:
        imgColour = np.dstack((img,img,img))
        
    elif len(size) == 3:
        if size[2] == 3:
            #print("Image already has 3 Channels")
            imgColour = img
        else:
            raise Exception(f"Input image has the incorrect amount of channels\nGot: {size[2]}, Expected 3")
    else:
        raise Exception(f"Input image has the incorrect dimensions\nGot: {len(size)}, Expected 2")

    return(imgColour)

def convertToGrey(img):
    pass

def stackImages(imgs:list, ratio:float = 9/16):

    def getFactorPairs(val):
        return [(i, int(val / i)) for i in range(1, int(val**0.5)+1) if val % i == 0]

    def closest_value(inputArray, test):
        i = (np.abs(inputArray - test)).argmin()
        return inputArray[i], i

    
    factors = getFactorPairs(len(imgs))

    if len(factors) == 1:
        filler = np.zeros(np.shape(imgs[0]))
        if len(np.shape(imgs[0])) > 2:
            filler = convertTo3Channel(filler)
        imgs.append(filler)
        factors = getFactorPairs(len(imgs))

    imgsArray = np.asarray(imgs)
    ratios1 = np.asarray([val[0]/val[1] for val in factors])
    ratios2 = np.asarray([val[1]/val[0] for val in factors])
    
    ratio1, ratioIndex1 = closest_value(ratios1, ratio)
    ratio2, ratioIndex2 = closest_value(ratios2, ratio)

    if ratio1 < ratio2:
        arrangement = factors[ratioIndex1]
    else:
        arrangement = factors[ratioIndex2]

    imageColumns = np.split(imgsArray, arrangement[1])
    vertImages = []
    for imageCols in imageColumns:
        vImage = imageCols[0,:,:]
        if np.shape(imageCols)[0] > 1:
            for i in np.arange(1, np.shape(imageCols)[0]):
                vImage = np.vstack((vImage, imageCols[i,:,:]))
        vertImages.append(vImage)

    outImage = vertImages[0]
    if len(vertImages) > 1:
        for i in np.arange(1, len(vertImages)):
            outImage = np.hstack((outImage, vertImages[i]))

    return outImage

def plot3d(img):
    fig = plt.figure()
    ha = fig.add_subplot(111, projection='3d')
    xRange = np.arange(np.shape(img)[0])
    yRange = np.arange(np.shape(img)[1])
    X, Y = np.meshgrid(yRange, xRange)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, img, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    plt.show()
    
def fourierFilter(img, lowpass:int = 0, highpass:int = 0, inv:bool = False):
    imgSize = np.shape(img)
    filterMask = np.asarray(np.zeros(imgSize),dtype=np.bool8)
    lowpassMask = np.zeros(imgSize)
    highpassMask = np.zeros(imgSize)

    centre3 = np.asarray(np.divide(imgSize, 2),dtype=np.int16)
    centre2 = np.subtract(centre3, (0,1))
    centre1 = np.subtract(centre3, (1,0))
    centre0 = np.subtract(centre3, (1,1))
    centresList = [centre0]
    if not centre3[0] % 2:
        #print(f"appending {centre1}")
        centresList.append(centre1)

    if not centre3[1] % 2:
        #print(f"appending {centre2}")
        centresList.append(centre2)

    if not centre3[0] % 2 and not centre3[1] % 2:
        #print(f"appending {centre3}")
        centresList.append(centre3)
    
    for centre in centresList:
        if lowpass:
            cv.circle(lowpassMask, centre, lowpass, 1, -1)
        if highpass:
            cv.circle(highpassMask, centre, highpass, 1, -1)

    if lowpass:
        lowpassMask = np.asarray(lowpassMask, dtype=np.bool8)
        filterMask = np.bitwise_or(filterMask, lowpassMask)

    if highpass:
        highpassMask = np.asarray(highpassMask, dtype=np.bool8)
        highpassMask = np.bitwise_not(highpassMask)
        filterMask = np.bitwise_or(filterMask, highpassMask)

    if inv:
        filterMask = np.bitwise_not(filterMask)

    filteredImg = np.multiply(img, filterMask)
    
    return filteredImg, filterMask

def edgeDetector(img):
    #blur = conv2D(img, gaussian)
    edgeX = conv2D(img, sobelX)
    edgeY = conv2D(img, sobelY)
    gradient = np.hypot(edgeX, edgeY)
    angle = np.arctan2(edgeX, edgeY)
    return gradient

def imageLoader(imgNo):
    directory = Path("COMP6001-A2-Datasets\\Frames-based Dataset")
    images = os.listdir(directory)
    if imgNo < len(images):
        image = Path(directory, images[imgNo])
    else:
        raise Exception(f"Image {imgNo} doesn't exist in directory\n Directory ranges from 0 to {len(images)-1}")

    img = cv.imread(str(image), cv.IMREAD_GRAYSCALE)
    imgColour = cv.imread(str(image))
    return img, imgColour

def superpixelMapping(img:np.ndarray, blur:int, checkRange:int):
    img = convertTo8Bit(img)
    size = np.shape(img)
    
    if len(size) <= 2:
        imgColour = np.dstack((img,img,img))
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgColour = img
    
    outputImg = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    count = 0
    superPixelList = []

    for row in np.arange(np.shape(img)[0]):
        for column in np.arange(np.shape(img)[1]):
            if not count % 100:
                print(f" Super Pixels Found: {len(superPixelList)}", end="\r")
            count += 1
            pixel = (row,column,img[row,column])
            if not superPixelList:
                newSuperPixel = superpixel(pixel, blur, checkRange)
                superPixelList.append(newSuperPixel)
            else:
                check = False
                for superPixel in superPixelList:
                    check = superPixel.checkpixel(pixel)
                    if check:
                        superPixel.addpixel(pixel)
                        break
                
                if not check:
                    newSuperPixel = superpixel(pixel, blur, checkRange)
                    superPixelList.append(newSuperPixel)

    pixelColour = np.random.randint(256, size=(3,len(superPixelList)))

    for superPixel in range(len(superPixelList)):

        superPixelList[superPixel].assignColour(pixelColour[:,superPixel])
        outputImg = np.add(outputImg, superPixelList[superPixel].generateSuperPixelArray(size))   
    outputImg = cv.addWeighted(imgColour,0.7,outputImg,0.3,0)
    print("\n")
    return superPixelList, outputImg

def labelImages(img, label:str):
    height = np.shape(img)[1]
    outImg = cv.putText(img, label, (10,height-10), cv.FONT_HERSHEY_SIMPLEX, .8, (255,255,255))
    return outImg
    
def houghCircles(img, dp:int= 1, minDist:int = 0, param1:float = 0, param2:float = 0, minRadius:int = 1, maxRadius:int = 255):
    img = convertTo8Bit(img)
    circles = cv.HoughCircles(img,method=cv.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    circleImage = makeCircles(img, circles)
    return circles, circleImage

def makeCircles(img, circles):
    img = convertTo8Bit(img)
    circleImage = convertTo3Channel(img)
    if circles is None:
        pass
    else:
        for i in circles[0,:]:
            circleImage = cv.circle(circleImage,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),1)
    return circleImage

def superPixelClassifier(img, superPixels:list, largeThreshold:int = 1000):
    labels = []
    mask = []
    for superPixel in superPixels:
        if superPixel.getPixelIntensity() > 122:
            if np.shape(superPixel.getSuperPixelData())[0] > largeThreshold:
                print("Found a large signal")
                superPixel.setClassification("Large Object")
                labelledMask, unlabelledMask = superPixel.generateMask(np.shape(img))
                labels.append((superPixel.getClassification(),labelledMask))
                mask.append(labelledMask)
                
        else:
            if np.shape(superPixel.getSuperPixelData())[0] > largeThreshold:
                print("Found Background")
                superPixel.setClassification("Background")
                labelledMask, unlabelledMask = superPixel.generateMask(np.shape(img))
                labels.append((superPixel.getClassification(),labelledMask))
                mask.append(labelledMask)
            
    labelMask = np.sum(np.asarray(mask), axis=0, dtype=np.bool8)
    labeledImg = np.multiply(img, labelMask, dtype=np.uint8)
    unlabeledMask = np.bitwise_not(labelMask)
    unlabeledImg = np.multiply(img, unlabeledMask, dtype=np.uint8)
    labels.append(("Unlabelled", unlabeledMask))
    
    return labeledImg, unlabeledImg, superPixels, labels
          
def houghSuperPixelErrorChecker(img, superPixelList, circles, allowedError:float):
    errorList = []
    circleList = []
    if circles is None:
        pass
    else:
        for circle in circles[0,:]:
            blank = np.zeros(np.shape(img))
            circleMask = cv.circle(blank,(int(circle[0]),int(circle[1])),int(circle[2]),1,-1)
            for superPixel in superPixelList:
                pixelMap = superPixel.generateMask(np.shape(img))[0]
                if pixelMap[int(circle[1]),int(circle[0])] == 1:
                    pixelMap = np.asarray(pixelMap,dtype=np.float64)
                    errorCheck = np.add(pixelMap, circleMask)

                    total = np.count_nonzero(errorCheck)

                    overlap = np.divide(np.sum(errorCheck[errorCheck > 1]),2)
                    error = np.divide(np.abs(np.subtract(total,overlap)),total)
                    underlap = np.sum(errorCheck[errorCheck == 1])
                    if error <= allowedError:
                        circleList.append(circle)
                        errorList.append((error, overlap, underlap, errorCheck))
    
    return circleList, errorList
        

    

if __name__ == "__main__":

    start = time()
    imageNo = 19
    img, imgColour = imageLoader(imageNo)

    # Original Image
    original = img

    # Fourier Transform Lowpass
    fourierTest = np.fft.fft2(img)
    filtered, mask = fourierFilter(fourierTest,lowpass=69, highpass=0, inv = False)
    inverserFourierTest = np.fft.ifft2(filtered)
    imageReconstruction = np.abs(inverserFourierTest)

    imageReconstruction = np.multiply(imageReconstruction, np.ones(np.shape(imageReconstruction), dtype=np.uint8))
    rescannedImage = np.multiply(imageReconstruction, img)

    # Convolution Edge Detection
    convolvedOriginal = edgeDetector(img)

    convolvedReconstruction = edgeDetector(imageReconstruction)

    convolvedRescanImage = edgeDetector(rescannedImage)
   
    
    
    # Super Pixel Generation
    superPixelList, superPixelTest = superpixelMapping(original, blur=12, checkRange=2)

    #Hough Circles
    circlesOriginal, circleOriginalImage = houghCircles(original, dp=1, minDist=8, param1=80, param2=5, minRadius=0, maxRadius=8)
    circlesLP, circleLPImage = houghCircles(imageReconstruction, dp=1, minDist=8, param1=80, param2=5, minRadius=0, maxRadius=8)
    circlesRS, circleRSImage = houghCircles(rescannedImage, dp=1, minDist=8, param1=80, param2=5, minRadius=0, maxRadius=8)
    circlesConv, circleConvImage = houghCircles(convolvedRescanImage, dp=1, minDist=8, param1=80, param2=5, minRadius=0, maxRadius=8)
    
    originalCircleImage = makeCircles(original, circlesOriginal)
    lowpassCircleImage = makeCircles(original, circlesLP)
    rescanCircleImage = makeCircles(original, circlesRS)
    convCircleImage = makeCircles(original, circlesConv)

    circleList, errorList = houghSuperPixelErrorChecker(original, superPixelList, circlesConv, allowedError=0.4)
    if circleList:
        circlesSP = np.reshape(circleList,(1,np.shape(circleList)[0],np.shape(circleList)[1]))
        errorCircleImage = makeCircles(original, circlesSP)
    else:
        circlesSP = []
        errorCircleImage = makeCircles(original, None)

    #labelledImg, unlabelledImg, superPixelList, labels = superPixelClassifier(original, superPixelList, largeThreshold = 1000)


    print(np.shape(circlesOriginal))
    print(np.shape(circlesLP))
    print(np.shape(circlesRS))
    print(np.shape(circlesConv))
    print(np.shape(circlesSP))


    #Image Dictionary
    imgs = {
        "Original":originalCircleImage, 
        "Lowpass":lowpassCircleImage, 
        "Lowpass Rescan":rescanCircleImage, 
        "Edges LP RS":convCircleImage, 
        "SP Corrected":errorCircleImage, 
    }

    # Image manipulation to make it clean to display
    imgs = {key: normalize(ims) for key, ims in imgs.items()}
    imgs = {key: convertTo3Channel(ims) for key, ims in imgs.items()}
    imgs = {key: rescaleImage(ims, 3) for key, ims in imgs.items()}
    imgs = {key: labelImages(ims, key) for key, ims in imgs.items()}

    # Image output stacker optimised to fit a 16:9 display
    imageOutput = stackImages(list(imgs.values()), ratio = 1/5)
    
    # Timer
    end = time()
    print(end-start) 

    cv.imshow("image",imageOutput)
    cv.waitKey(0)
    cv.destroyAllWindows()