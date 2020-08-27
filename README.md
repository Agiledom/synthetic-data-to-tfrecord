## Synthetic data to tfrecord

##### An overview

This tool is intended for use with Tensorflow's Object Detection API. Its intention is to assist people who
wish to detect objects that are not currently part of an open-source data set (i.e. COCO), and whom struggle to find or
label the hundreds or thousands or images it can take to train an efficient Object Detection Model.

The tool contains the following core scripts: 
1. generate_dataset.py - takes a series of backgrounds, along with a series of objects and generates synthetic images,
along with the requisite annotations (bounding boxes, image size etc.) required to create tfrecords.

2. generate_tfrecord.py - takes in the images and annotations and serialises them into tfrecords, ready for training
and evaluation with a TF2 model.

3. clean_image.py - takes in an object image with a background and uses canny edge detection to output an object image
with a transparent background (a prerequisite for generate_dataset.py).

#### Instructions

##### Step 1 - Installing the Object Detection API

You have two options when using this tool, you can either run it locally or you can run it on Google Cloud Platform 
(GCP). Both options require you to install Tensorflow 2's Object Detection API. Instructions on how to do this, can be
found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

If you would like a GCP with the Object Detection API pre-installed, you can use this repository here:
https://github.com/Agiledom/TF2-ObjectDetection-VM. This tool will also preset the permissions required by GCP for you
to read and write to cloud storage buckets, from your GCP VM.

##### Step 2 - Getting ready for liftoff.

*Clone the repository*

    cd models/research
    git clone https://github.com/Agiledom/synthetic-data-to-tfrecord.git

*Upload your data*

 If you are running the script locally, copy and paste your backgrounds into the input/backgrounds directory and your
 objects into the input/objects directory.
 
 If you are running the script from within a GCP VM, you need to create a bucket in GCP Storage along with the
 directories input/backgrounds and input/objects. Then upload your backgrounds and objects into the requisite
 directories. 
 
 ##### *Quick side bar - some important notes about the input images.*
 
 1. You can utilise this tool for images of any size. However I recommend using background images that correlate to the
 model you intend to use. For a list of TF2 Object Detection models, see:
 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
 
 2. The object images should be in a png format, with a transparent background. If you have images that need "cleaning",
 you can utilise the clean_image.py tool (see below).

 3. The object images should be a least half the width and height of the size of the background images. Eg. for a
 500 x 500 background, the object images should be no larger than 250 x 250. This is important for the group stage
 in generate_dataset.py. If the object images are too big, you will get a numpy error.
 
 4. Line 18 of generate_tfrecord.py sets the a variable that determines the number of images that will go into each
 tfrecord shard. One size doesn't fit all here. If you are using larger images, you should reduce this number, and vice
 versa for smaller images. Ideally each shard should be between 100-300mb in size.
 
 ##### Step 3 - Liftoff
 
    cd synthtetic-data-to-tfrecord
 
If you are running on a GCP VM, run:
 
    # add your bucket name to the environment variables
    # do NOT include "gs://"
    pip3 install -r requirements.txt
    export BUCKET=<your-awesome-bucket-name>
    python3 main.py -cld True
    
Else:

    pip3 install -r requirements.txt
    python3 main.py
    
##### clean_image.py

If you have images that have a background that you want to remove to produce the transparent png the tool requires,
you can place these images into the clean_image directory. Then run:
       
    python3 clean_image.py
    
The images will be outputted into the input/objects directory.
 
 ##### Credits + useful links
 
 Tyler Hutcherson - for providing the bulk of the generate_dataset.py code that this tool was built on top of and for
 writing this awesome article:
 https://medium.com/skafosai/generating-training-images-for-object-detection-models-8a74cf5e882f
 
 Datitran - for his work here: https://github.com/datitran/raccoon_dataset
 
 Sentdex - pythonprogramming.net for his step by step Object Detection Tutorial:
 https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/
 
 ##### Call to arms
 
 If anyone would like to get involved with expanding this either vertically (to add more useful supporting features for
 the Object Detection API) or laterally to encompass other research models, please do a pull request or get in touch.