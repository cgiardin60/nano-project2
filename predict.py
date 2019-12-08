# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:06:29 2019
This program is part of the Udacity nano - Deep Learning project 
Build a Command Line version of the Image Classifier
This program leverages previously created model to predict an image class
@author: CHRISTOPHERGiardina
"""

#first add all imports required

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import sys
import os.path
import json
# in this program we'll add the argparse directly instad of using separate python file
import argparse
#######################################################################################
"""
THIS SECTION CONTAINS FUNCTIONS
"""
#######################################################################################
def allparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type = str,
                        help="STR - The full path and name to the image to be processed")
    parser.add_argument('chkp_file',type=str, 
                        help="STR - The full path and file name of the trained 'MODEL CHECKPOINT' file.  e.g. c:\Python\model.pth")

    parser.add_argument('--normal_means',type=str, 
                        help = "STR - three MEAN values for normalizing images, string converted to 3 FLOAT values in code. default = \"0.485, 0.456, 0.406\"")

    parser.add_argument('--normal_stds',type=str, 
                        help = "STR - three STD DEVIATIONS for normalizing images, string converted to 3 FLOAT values in code default = \"0.229, 0.224, 0.225\"")

    parser.add_argument('--cat_to_name',type=str, 
                        help = "STR - Optional -  full path and file name of a JSON file that maps the CLASS Number to the english CLASS NAME." )

    parser.add_argument('--device',type=str, 
                        help = "STR either \"cuda\" to run on a GPU or \"cpu\" to run on CPU. Turns on GPU for model processing default is cpu") 
      
    parser.add_argument('--top_k', 
                        type=int, 
                        help='INT - The top \'n\' probabilities you wish to see for image class predictions -- default = 5 ')
    args = parser.parse_args()
    return args

# Parse args
def open_file(filenm):
    rtn_tensor=torch.tensor
    
    try:
        infer_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img_pil = Image.open(filenm)
        img_tensor = infer_transform(img_pil)
        rtn_tensor =  img_tensor
    except:
        print ("error occurred in open file function", sys.exc_info()[0])
        
    return rtn_tensor
        
def load_checkpoint(filepath):
    #load in checkpoint data to dictionary
    checkpoint = torch.load(filepath)
    #--parse name
    
    model_nm = checkpoint['ntwk_arch']
    #set our model type
    str = "models." + model_nm + "(pretrained=True)"

    model = eval(str)
    #freeze params for the features (sequential) layers
    for param in model.parameters():
        param.requires_grad = False
    
    #Now load up other custom things from saved model 
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])

    #for other items in the checkpoint- like epochs and optimizer state
    #these can be addressed once model returned 
    
    model.model_name = model_nm
    return model    

def predict(file_tensor, model, device, topk):
    
    # TODO: Implement the code to predict the class from an image file
    # No need for GPU 
    model.to(device);
    #print(device)
    
    # Set model to evaluate
    model.eval();
    
    #this requires some reshaping to get  our needed tensor shape of [1,3,224,224]
    tensor_image = F.interpolate(file_tensor.unsqueeze(0),scale_factor=1)
    #then we take to correct device - in this case 'cpu'
    tensor_image = tensor_image.to(device);

    # TODO: Calculate the class probabilities (softmax) for img
    ps = torch.exp(model(tensor_image))
    top_p, top_index = ps.topk(topk, dim=1)
    
    #We want to return four (4) return vales if possible
    # 1 - top probabilities
    # 2 - the top associated model result indexes
    # 3 - the actual Image Class codes - that map to the Indexes
    # 4 - the Image Names  - if these are provided in a json file on the CLI aruments 
    
    # flatten the tensors first 
    top_p_flat = [element.item() for element in top_p.flatten()]
    top_index_flat = [element.item() for element in top_index.flatten()]
    
    # also need the dictionary of indexes -to- image file class codes 
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    #Then check for a cat_to_json file
    catfile = True if allargs.cat_to_name else False
    
    #Now we need a list of class_ids
    class_ids = list()
    for idx in top_index_flat:
        class_ids.append(idx_to_class[idx])
    
    #now want to build the images names if a cat_to_names file was provided
    top_classes = {}
    if catfile:
        with open(allargs.cat_to_name, 'r') as f:
            catnames = json.load(f)
        for clsstr in class_ids:
            top_classes.update({clsstr: catnames[str(clsstr)]})
            
    #round probabilities
    
    #returning the top probs and indexes as Lists -  as described in instructions, and Class List + flowers as a dictionary 
    #return [str(round(x*100,4)) + '%' for x in top_p_flat], top_index_flat, class_ids, top_classes
    return ["{:.2%}".format(x) for x in top_p_flat], top_index_flat, class_ids, top_classes
#######################################################################################
"""
THIS SECTION CONTAINS MAIN PROGRAM FLOW
"""
#######################################################################################
allargs = allparser()
#Get the file - use a DEF
filenm = allargs.file_path

if os.path.isfile(filenm):
    file_tensor = open_file(filenm)
else:
    #bad checkpoint file passed in
    print("Can't find image file, double check location or spelling")
    exit

#print(file_tensor.shape)


#Get the model - from checkpoint file and fix it up - USE a DEF
#--check for file
if os.path.isfile(allargs.chkp_file):
    chkpnt_file = allargs.chkp_file
else:
    #bad checkpoint file passed in
    print("Can't find model checkpoint file, double check location or spelling")
    exit
# load checkpoint
# reconstituting model using model_name param
# and set all proper checkpoint parts
testing_model = load_checkpoint(chkpnt_file)
#print(testing_model)
print("Using Torchvision model:",testing_model.model_name)

# Get device to run on
# set default first
test_device = torch.device('cpu')
dvc = allargs.device
if dvc != None: 
    if dvc == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            test_device = torch.device('cuda')
        else:
            test_device = torch.device('cpu')

print("Using device:", test_device)

# Get top_k
t = lambda x: 5 if x == None else x
t_k = t(allargs.top_k)

print("Listing top", t_k, " probabilities")

# run Prediction and return outputs
probs, indexes, classes, flowerlist = predict(file_tensor, testing_model, test_device, t_k)

#print the prediction

print ("Top Probabilities: ", probs)
print('Top Model Indexes:', indexes)
print('Top Flower Class Ids:', classes)
print('Top Class Ids with Flower Names:',flowerlist)

