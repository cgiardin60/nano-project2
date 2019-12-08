# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 09:49:29 2019

@author: CHRISTOPHERGiardina
"""

import argparse
def allparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type = str,
                        help="STR - the main file system folder where data files to be classified live - assumes organized into three subfolders : \\train, \\valid, \\test")

    parser.add_argument("out_class", type = int,
                        help="INT - the specific number of OUTPUT CLASSES in the OUTPUT Layer - this depends on images classifying")

    parser.add_argument('chkp_dir',type=str, 
                        help="STR - filesystem path for saving the trained 'MODEL CHECKPOINT' file- e.g. -  c:\Python")

    parser.add_argument('--ntwk_arch',type=str, 
                        help = "STR - specific NETWORK ARCHITECTURE model from torchvison.models to be used - default = \"vgg16\"")

    parser.add_argument('--normal_means',type=str, 
                        help = "STR - three MEAN values for normalizing images, string converted to 3 FLOAT values in code. default = \"0.485, 0.456, 0.406\"")

    parser.add_argument('--normal_stds',type=str, 
                        help = "STR - three STD DEVIATIONS for normalizing images, string converted to 3 FLOAT values in code default = \"0.229, 0.224, 0.225\"")

    parser.add_argument('--batch_size',type=int, 
                        help = "INT - batch size for data loaders -- default = 72")

    parser.add_argument('--device',type=str, 
                        help = "STR - either \"cuda\" to run on a GPU or \"cpu\" to run on CPU. Default is \"cuda\" to run on GPU device") 
      
    parser.add_argument('--learn_rate', 
                        type=float, 
                        help='FLOAT - Define optimizer learning rate -- default = 0.003 ')
    
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='INT  - Hidden units for classifier -- default = 612')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        help='INT - Number of epochs for training -- default = 2')


# Parse args
    args = parser.parse_args()
    return args
