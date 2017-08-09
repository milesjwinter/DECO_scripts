# DECO_scripts
Various analysis scripts for DECO

## Classify_plotBlobs.py
Loads DECO images from an input directory and displays each blob above threshold. The user inputs the classification for each blob and the results are written to a text file that can be passed to make_gray_image_database.py. 
## DECO_CNN.py
For constructing and training the CNN
## Predict_Label_plotBlobs.py 
Same idea as Classify_plotBlobs.py, but the CNN prediction is available for each blob.
## make_gray_image_database.py 
Generate an hd5f database of images and labels that can be used to train the CNN. 
## old_zoom_predictions.py     
Base script for applying CNN predictions to new DECO images
## reverse_lookup.py
Finds the country of origin for DECO events based on metadata 
## rotated_dcgan_deco.py
Implemention of DCGAN in TFLearn using data augmentation to generate images from a rotation invariant dataset. 
## zoom_predictions.py
Updates script for applying CNN predictions to new DECO images. Designed to integrated with the nightly processing pipeline. 
