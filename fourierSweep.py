import numpy as np
import cv2 as cv
import time
import imageDataProcesser as imProcess


windowName = "Frequency Response" # Initial Parameters for open cv window display

imageNo = 14
img, imgColour = imProcess.imageLoader(imageNo)

cv.namedWindow(windowName, 1)

for frequency in range(1,75):

    fourierTest = np.fft.fft2(img)
    filtered, mask = imProcess.fourierFilter(fourierTest,lowpass= frequency)
    inverserFourierTest = np.fft.ifft2(filtered)
    imageReconstruction = imProcess.normalize(np.abs(inverserFourierTest))
    imageReconstruction = imProcess.rescaleImage(imageReconstruction, 4)

    cv.imshow(windowName, imageReconstruction)
    time.sleep(0.1)
    c = cv.waitKey(1) & 0xFF
    if c == ord("q"):
        break


cv.destroyAllWindows()
c= cv.waitKey(1)

