# -*- coding: utf-8 -*-
"""
Created on Wed Dec 4 10:06:19 2019
This program is part of the Udacity nano - Deep Learning project 
Build a Command Line version of the Image Classifier
This program defines and trains the models and saves it to a checkpoint file
@author: CHRISTOPHERGiardina
"""

#first add all imports required
import numpy as np
import torch
from torch import nn
from torch import optim
#import torch.nn.functional as F
from torchvision import datasets, transforms, models
import sys
import time

# Next we establish the command line argument parser
# this has been externalized into a separate python file 'pyarg.py'
# this will create one dictionary 
import pyarg as AP

# test the allargs dictionary
#print(allargs.datapath)

##############################################################################
"""
section for Defined FUNCTIONS
"""
##############################################################################
def build_train_data(trndir,n_means,n_stds,s_batch):
    #do TRAIN transforms, datasets and loaders
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(n_means,
                                                            n_stds)])
    
    train_data = datasets.ImageFolder(trndir, transform=train_transforms)  
    trainload = torch.utils.data.DataLoader(train_data, batch_size=s_batch, shuffle=True)
    
    return trainload , train_data

def build_val_test_data(trndir,n_means,n_stds,s_batch):
    #do VALIDATE/TEST transforms, datasets and loaders
    val_test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(n_means,
                                                           n_stds)])
    
    val_test_data = datasets.ImageFolder(trndir, transform=val_test_transforms)
    
    valtestload = torch.utils.data.DataLoader(val_test_data, batch_size=s_batch, shuffle=True)
    
    return valtestload

def get_out_nodes(dif_model,netwkname):
    #this support using a different model than the default vgg16
    ntwk=netwkname[:3]
    if ntwk == 'ale':
        out_nodes = dif_model.classifier[1].in_features
    elif ntwk == 'vgg':
        out_nodes = dif_model.classifier[0].in_features
    elif ntwk == 'res':
        out_nodes = dif_model.fc.in_features
    elif ntwk == 'den':
        out_nodes = dif_model.classifier.in_features
    elif ntwk == 'squ':
        out_nodes = dif_model.classifier[1].in_channels
    elif ntwk == 'inc':
        out_nodes = dif_model.fc.in_features
    else:
        out_nodes = 25088 # assuming vgg16 network model
    
    return out_nodes

def define_model():
    #First check for any different than default vgg16 network type
    #and then get the features_out for Classifier_In nodes for first Linear step
    if allargs.ntwk_arch == None: 
        d_model = models.vgg16(pretrained=True)
        nodes_in = 25088
        n_name = 'vgg16'
    else:
        n_name = allargs.ntwk_arch.casefold()
        d_model = eval ("models." + n_name + "(pretrained = True)")
        nodes_in = get_out_nodes(d_model, n_name)
    d_model.name = n_name
        
    #freeze the params for base Sequential   
    for param in d_model.parameters():
        param.requires_grad = False
    
    #get out_class nodes number - this is a REQUIRED CLI arg
    out_cnodes = allargs.out_class
    
    #get hidden units if defined - optinal- default to 612
    if allargs.hidden_units == None: 
        h_units = 612
    else:
        h_units = allargs.hidden_units
    
    print("Classifier hidden units :", h_units)

    
    # redeine the model classifier with correct out/in nodes
    d_model.classifier = nn.Sequential(nn.Linear(nodes_in, h_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(h_units, out_cnodes),
                                 nn.LogSoftmax(dim=1))
    
    return d_model

def train_model(model, trn_val_device, trainloader, validloader, n_epochs, steps, running_loss, print_every, optimizer, criterion):
    try:
        for epoch in range(n_epochs):
        #using a simple try--> except here to demo how might add more robust error handling
            for images, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                images, labels = images.to(trn_val_device), labels.to(trn_val_device)
                
                #zero out gradients
                optimizer.zero_grad()
                
                #get forward pass probabilities, calc the loss and backprop to update weights 
                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
            
            # this check sees we hit 5 steps and then does validate check and printout of progress
                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    rtn = True
                    #set model to 'eval' as we don't want to update weights
                    model.eval()
                    with torch.no_grad():
                        for images, labels in validloader:
                            images, labels = images.to(trn_val_device), labels.to(trn_val_device)
                            logps = model(images)
                            batch_loss = criterion(logps, labels)
                            
                            valid_loss += batch_loss.item()
                            
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                    print(f"Epoch {epoch+1}/{n_epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validate loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validate accuracy %: {100 *(accuracy/len(validloader)):.3f}")
                    running_loss = 0
                    #reset model back to training
                    model.train()
                    rtn = True
    except:
        print ("error occurred in train_model function", sys.exc_info()[0])
        rtn = False
    return rtn

def test_model(model, trn_val_device,testloader, test_loss, accuracy, optimizer, criterion):
    #want no optimizing of weights
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(trn_val_device), labels.to(trn_val_device)
            logps = model(images)
            #test_loss = criterion(logps, labels)
    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test Accuracy: {accuracy/len(testloader)* 100:.3f} %")
    model.train()
    return True

##############################################################################
"""
SECTION  for MAIN processing 
"""
##############################################################################
# Get the arguments
allargs = AP.allparser()

#Set the data directory and sub folders
#using bespoke subfolders names - these are not hyper-params in this version
#Data dir is a REQUIRED CLI argument
data_dir =  allargs.data_path
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Call Train / Valid & Test transforms --> producing Data Loaders
# get any normalizing args and batch size provided in optional CLI args
m = lambda x: "0.485, 0.456, 0.406" if x == None else x
str_means = m(allargs.normal_means)
n_means = list(np.float_(str_means.split(',')))
print("Normalizing means values: ", n_means)

s = lambda x: "0.229, 0.224, 0.225" if x == None else x
str_stds = s(allargs.normal_stds) 
n_stds = list(np.float_(str_stds.split(',')))
print("Normalizing standard dev values: ", n_stds)

b = lambda x: 72 if x == None else x
s_batch = b(allargs.batch_size)
print("Batch size: ", s_batch)

#Call the data loader crate process
trainloader, training_data = build_train_data(train_dir, n_means,n_stds,s_batch)
validloader = build_val_test_data(valid_dir, n_means,n_stds,s_batch)
testloader = build_val_test_data(test_dir, n_means,n_stds,s_batch)

#Build Network Arch model#

#Call model define function
model = define_model()
print("Network Architetcure :" , model.name)

#Get learn rate and set Loss function and Optimizer an set
l = lambda x: 0.003 if x == None else x
l_rate = l(allargs.learn_rate)
print("Learning rate: ",l_rate )
    
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = l_rate)


# Get device to run on
# set model device from CLI args or use default of 'cuda' 
if allargs.device == None: 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        trn_val_device = torch.device('cuda')
    else:
        trn_val_device = torch.device('cpu')
else:
    if allargs.device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            trn_val_device = torch.device('cuda')
        else:
            trn_val_device = torch.device('cpu')

model.to(trn_val_device)
print("Device: ", trn_val_device)

# Execute Training & Validation Runs - that prints model training status along the way
# Set epochs from CLI args if present and set other variable intializations
e = lambda x: 2 if x == None else x
n_epochs = e(allargs.epochs)
print("Number of epochs: ", n_epochs)
steps = 0
running_loss = 0
print_every = 5
start = time.time()

print ("\nTraining model starting...")
m_train = train_model(model, trn_val_device, trainloader, validloader, n_epochs, steps, running_loss,print_every, optimizer, criterion)
print ("\nTraining model succeeded =", m_train)
print (f"Device = {trn_val_device}; Total Training Time: {(time.time() - start)/60:.3f} minutes")

#Clear gpu memory 
torch.cuda.empty_cache()

# Execute TEST Run - to verify model on untrained data
# use built model, device and testlaoder
test_loss = 0
accuracy = 0
print ("\n\nTesting of model starting...")
m_test = test_model(model, trn_val_device, testloader,test_loss, accuracy,optimizer, criterion)
print ("Testing model succeeded =", m_test)

#Clear gpu memory
torch.cuda.empty_cache()

#Save the model Checkpoint file
# first get the class to index
model.class_to_idx = training_data.class_to_idx
#get checkpoint file path/name -- this is a REQUIRED CLI arg
chkpnt_file = allargs.chkp_dir + "\model_" + model.name + "_checkpoint.pth" 


#set params
checkpoint = {'ntwk_arch': model.name,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'epochs': n_epochs}
#Save checkpoint
print ("\nStarting model checkpoint save...\n")
torch.save(checkpoint, chkpnt_file)
print("\nCheckpoint file saved at :",chkpnt_file )
