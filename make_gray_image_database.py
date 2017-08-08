import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpl
from matplotlib import gridspec
from PIL import Image
import h5py

#fix random seed for reproducibility
#seed = 8
#np.random.seed(seed)

def get_blobtype(x): #Returns number value assigned to each type
    return {1:'spot',0:'worm',3:'ambig',2:'track',5:'noise',6:'edge',7:'empty'}[x]

def get_numtype(x): #Returns number value assigned to each type
    return {'spot':1,'worm':0,'ambig':3,'track':2,'noise':5,'edge':6,'empty':7}[x]
vec_get_numtype = np.vectorize(get_numtype)

def get_binary_numtype(x): #Returns number value assigned to each type
    return {'spot':0,'worm':0,'ambig':3,'track':1,'noise':5,'edge':6,'empty':7}[x]
vec_get_binary_numtype = np.vectorize(get_binary_numtype)

def getZoomedBoundingBox(xavg, yavg, size=32):
    return xavg-size, xavg+size, yavg-size, yavg+size

def check_edges(xavg, yavg, xdim, ydim, crop_size=64): #checks if image is on the edge of the sensor
    x0, x1, y0, y1 = getZoomedBoundingBox(xavg,yavg,size=crop_size/2)
    results = np.array([x0>=0,x1<=xdim,y0>=0,y1<=ydim])
    return results.all()==True

#load initial blob type classifications
data = np.loadtxt('new_blob_labels.txt',dtype='S')
xavg = data[:,0].astype(int)
yavg = data[:,1].astype(int)
blob_type = data[:,2]
file_path = data[:,4]
blob_num = vec_get_numtype(blob_type)
#blob_num = vec_get_binary_numtype(blob_type)

#Specify image dimensions
dims = 64 #64
channels = 1 #1 for grayscale, 3 for rgb

#create database for holding images and labels
f = h5py.File("DCGAN_DECO_Image_Database_64.h5","w")
training_images = f.create_dataset("train/train_images", (1,dims,dims,channels), maxshape=(None,dims,dims,channels), dtype="uint8",chunks=True)
training_labels = f.create_dataset("train/train_labels", (1,), maxshape=(None,), dtype="uint8", chunks=True)

#Loop through all images for classifying purposes
random_index = np.random.permutation(len(xavg)).astype(int) 
for i in random_index:
    if int(blob_num[i])<4:
        img = Image.open(file_path[i]).convert('RGB')
        x0, x1, y0, y1 = getZoomedBoundingBox(xavg[i],img.size[1]-yavg[i],size=dims/2)
	cropped_img = np.array(img.crop((x0,y0,x1,y1)))
        if check_edges(xavg[i], img.size[1]-yavg[i], img.size[0], img.size[1], crop_size=dims)==True:
	    gray_img = np.mean(cropped_img,axis=2)
            gray_img = gray_img.reshape(dims,dims,channels)
	    training_images[-1] = gray_img
	    training_labels[-1] = int(blob_num[i])
	    print "Adding Image to Database: "
	    print file_path[i]
	    #resize database unless in final loop
	    if i != random_index[-1]:
		training_images.resize(training_images.shape[0]+1,axis=0)
		training_labels.resize(training_labels.shape[0]+1,axis=0)
        
        else:
            print "Skipping Image: on the edge of the camera sesnor"
            print file_path[i]
        print training_images.shape
f.close()
