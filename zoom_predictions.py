#!/usr/bin/env python

#------------------------------------------------------------------------#
#  Author: Miles Winter                                                  #
#  Date: 07-14-2017                                                      #
#  Project: DECO                                                         #
#  Desc: zoom in on brightest pixel and classify blobs with CNN          #
#  Note: Need the following installed:                                   #
#        $ pip install --user --upgrade h5py theano keras                #
#        Change keras backend to theano (default is tensorflow)          #
#        Importing keras generates a .json config file                   #
#        $ KERAS_BACKEND=theano python -c "from keras import backend"    #
#        Next, to change "backend": "tensorflow" -> "theano" type        #
#        $ sed -i 's/tensorflow/theano/g' $HOME/.keras/keras.json        #
#        Documentation at https://keras.io/backend/                      #
#------------------------------------------------------------------------#

try:
    import os
    import numpy as np
    import h5py
    import keras
    from keras.models import load_model
    from PIL import Image
except ImportError,e:
    print(e)
    raise SystemExit

def get_predicted_label(probs):
    """Returns predicted label. Track if prediction is > 60% """
    if probs[-1]>.6:
        return 'Track'
    else:
        return 'Background'

def get_crop_range(maxX, maxY, size=32):
    """define region of image to crop"""
    return maxX-size, maxX+size, maxY-size, maxY+size

def pass_edge_check(maxX, maxY, img_shape, crop_size=64):
    """checks if image is on the edge of the sensor"""
    x0, x1, y0, y1 = get_crop_range(maxX,maxY,size=crop_size/2)
    checks = np.array([x0>=0,x1<=img_shape[0],y0>=0,y1<=img_shape[1]])
    return checks.all()==True

def convert_image(img,dims=64,channels=1):
    """convert image to grayscale, normalize, and reshape"""
    img = np.array(img,dtype='float32')
    gray_norm_img = np.mean(img/255.,axis=-1)
    return gray_norm_img.reshape(1,dims,dims,channels)

def get_brightest_pixel(img):
    """get brightest image pixel indices"""
    img = np.array(img)
    summed_img = np.sum(img,axis=-1)
    return np.unravel_index(summed_img.argmax(), summed_img.shape)

def run_blob_classifier(paths, outfile):
    """classify blobs with CNN"""
    #Load CNN model
    try:
        model = load_model('trained_DECO_CNN.h5')
    except IOError:
        print('model could not be loaded...')
        raise SystemExit
    
    #load outfile
    f_out = ''
    if os.path.isfile(outfile):
        f_out = open(outfile,'a')
    else:
        f_out = open(outfile,'w')
        f_out.write("predicted_label,probability,filename")
        f_out.write("\n")

    #Loop through all images in the paths list
    for filename in paths:

	#load image
	try:
	    image = Image.open(filename).convert('RGB')
	except IOError,e:
	    print(e)
	    break

	#find the brightest pixel
	maxY, maxX = get_brightest_pixel(image)

	predicted_label = ''
	probability = 0.
	#check if blob is near sensor edge
	if pass_edge_check(maxX, maxY, image.size)==True:
	    #crop image around the brightest pixel
	    x0, x1, y0, y1 = get_crop_range(maxX,maxY)
	    cropped_img = image.crop((x0,y0,x1,y1))
	    
	    #Convert to grayscale, normalize, and reshape
	    gray_image = convert_image(cropped_img)

	    #predict image classification
	    probability = model.predict(gray_image, batch_size=1, verbose=0)
	    probability = probability.reshape(3,)

	    #convert prediction probability to a single label
	    predicted_label = get_predicted_label(probability)
	else:
	    probability = np.array([-1]*3)
	    predicted_label = 'Edge'
	    
        f_out.write("{},{},{}".format(predicted_label, probability, filename))
        f_out.write("\n")
    f_out.close()
