<<<<<<< HEAD
# Udacity Nanodegree Program - Introduction to Machine Learning : Project II- build an Image Classifier

This repo contains code for my final submission for the Udacity's AI Programming with Python Nanodegree program- Project II, IMage Classifier.

 In this project, an image classifier is built first in a Jupyter notebook with  PyTorch, then it is converted to  command line application - for both building /training a model, and then predicting the type of  (flower ) image in an image firel.

## Files included: (all Python v.3) ##
 1. Completed  jupyter notebook : Image Classifier project.ipynb
 2. HTML version of this completed notebook: Image Classifier Project.html
 3. The checkpiont saved vgg16 model created in the notebook: mymodelcheckpoint.pth - 
 4. Completed train.py python CLI executable program - takes 3 mandatory & 8 optional CLI arguments to build and train a model using torchvision models as base
 5. Complementary pyarg.py file to train.py that contains all the argprase managed command line arguments
 6. A helper python file- borrwed from  the Intro to ML training that helps with plotting visuals: helper.py
 7. An Example checkpoint files - for saved model (written from train.py):

### A vgg16 model with following arguments: model_vgg16_checkpoint.pth
 	 	Normalizing means values:  [0.485, 0.456, 0.406]
		Normalizing standard dev values:  [0.229, 0.224, 0.225]
		Batch size:  72
		Classifier hidden units : 612
		Network Architetcure : vgg16
		Learning rate:  0.002
		Device:  cuda
		Number of epochs:  2
 8. Completed predict.py python CLI executable program - takes 2 mandatory & 5 optional CLI arguments  -  predicts an image class when provided checkpoint file and image files (path + name)
 9. The image classID-to-image class name file used in building project deliverables for the project from the  102 Category Flower Dataset, Maria-Elena Nilsback and Andrew Zisserman (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) : cat_to_name.json
 10. The /assets  folder that includes a couple images for the Notebook 
 11. This README.md file
=======
