
from time import sleep
import scipy.io
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
import cv2 as cv
import matplotlib.pyplot as plt


dtype = np.dtype([("x", '<u2'), ("y", '<u2'), ('t', '<u8')])

# Event Data Loader
def dataloader(filename):
    mat = scipy.io.loadmat(filename) # x y ts

    
    dataset = []
    for datachunk in mat['spike_data_cell_array']:
        temp = np.stack((datachunk[0][0][0][0][0], datachunk[0][0][0][1][0], datachunk[0][0][0][2][0]))
        temp = np.transpose(temp) #Transpose array
        temp = unstructured_to_structured(temp, dtype=dtype) # Fit array to structured array
        dataset.append(temp)
    return dataset

# Data Visualisation
def visualiser(dataset, simulationTime:float = 0.1, scale:int = 4):
    if len(np.shape(dataset)) == 3:
        #3D point cloud
        t = np.arange(np.shape(dataset[0,0,:])[0])
        eventData = False
        pass
    else:
        #Event Data
        t = np.arange(0, np.max(dataset["t"])+1)
        surface = np.zeros((np.max(dataset["x"])+1,np.max(dataset["y"])+1))
        eventData = True
        
    
    windowName = "Dataset analysis" # Initial Parameters for open cv window display
    cv.namedWindow(windowName, 1)
    
    for i in t:
        if eventData:
            coords = dataset[dataset["t"] == i]
            surface[coords["x"],coords["y"]] = 255
        else:
            surface = reconstructedData[:,:,i] * 255
        tempImg = surface
        tempImg = np.repeat(tempImg, scale, axis=0)
        tempImg = np.repeat(tempImg, scale, axis=1)
        print(" data at t = %s"%(i), end="\r")
        disp = np.asarray(tempImg, dtype= np.uint8)
        cv.imshow(windowName, disp)
        sleep(simulationTime)
        surface[:,:] = 0
        c = cv.waitKey(1) & 0xFF
        if c == ord("q"):
            break
    cv.destroyAllWindows()
    c= cv.waitKey(1)

# Fourier Lowpass filter
def lowpassFilter(fourierArray, cutoff):
    
    lowpassFilter =  np.ones(cutoff)
    lowpassExcess = np.zeros(np.shape(fourierArray[0,0,:])[0]-np.shape(lowpassFilter)[0])
    lowpassFilter = np.hstack((lowpassFilter, lowpassExcess))

    filteredFourier = np.multiply(fourierArray, lowpassFilter)

    return(filteredFourier)

# Convert event data to 3D representation
def pointCloudRepresentation(dataset):
    x = np.asarray(dataset["x"],dtype=np.uint8)
    y = np.asarray(dataset["y"],dtype=np.uint8)
    t = np.asarray(dataset["t"],dtype=np.uint8)
    xAxis = x.max()+1
    yAxis = y.max()+1
    tAxis = t.max()+1
    representation3D = np.zeros((xAxis, yAxis, tAxis))
    representation3D[x,y,t] = 1
    return representation3D

# Convert 3D representation to Event data
def eventRepresentation(pointCloud):
    convertedEvents = np.where(pointCloud == 1)
    convertedEvents = np.stack(convertedEvents)
    convertedEvents = np.transpose(convertedEvents)
    convertedEvents = unstructured_to_structured(convertedEvents, dtype=dtype)
    return convertedEvents


if __name__ == "__main__":
    # Initialise data and load into system
    filename = "COMP6001-A2-Datasets\\Event-based Dataset\\student_1_spike_data.mat"
    dataset = dataloader(filename)

    # Choose dataset to view
    dataOpt = 6
    activeSet = dataset[dataOpt]

    # INDEX SHOULD START AT 0!
    activeSet["x"] = activeSet["x"]-1
    activeSet["y"] = activeSet["y"]-1

    # Generate 3D point cloud representation
    representation3D = pointCloudRepresentation(activeSet)

    # Compute the fourier transform across the time axis
    fourier3D = np.fft.fft(representation3D, axis=2)
    # Lowpass the fourier transform representation
    filteredFourier = lowpassFilter(fourier3D, cutoff = 8)
    # Return the data to the time domain get the absolute value and round to whole integer
    reconstructedData = np.round(np.abs(np.fft.ifft(filteredFourier, axis=2)))

    # Generate Event Data List
    filteredSet = eventRepresentation(reconstructedData)

    centreList = []
    for t in np.arange(filteredSet["t"].max()):
        slice = filteredSet[filteredSet["t"]==t]
        xCentre = np.mean(slice["x"])
        yCentre = np.mean(slice["y"])
        centreList.append((xCentre, yCentre, t))
    centreData = np.asarray(centreList, dtype=dtype)
    
    # Get Noise Events out of point cloud representation
    noise3D = representation3D
    noise3D[filteredSet["x"], filteredSet["y"], filteredSet["t"]] = 0
    noise = eventRepresentation(noise3D)

    # Plot source and noise point clouds
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    labeled = ax.scatter(filteredSet["x"], filteredSet["y"], filteredSet["t"],marker='^',alpha=0.1)
    centre = ax.plot(centreData["x"], centreData["y"], centreData["t"], color="green")
    noiseData = ax.scatter(noise["x"], noise["y"], noise["t"], marker='o',alpha=0.01)
    ax.legend([labeled, noiseData], ['Source', 'Noise'])
    plt.show()

    # Visualise Raw Data
    # visualiser(activeSet, simulationTime=0.1, scale=4)

    # Visualise Filtered Data
    # visualiser(reconstructedData)
    

