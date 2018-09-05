#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:43:32 2018

@author: abhijith
"""

import pickle
from sklearn.model_selection import train_test_split
from scipy import misc
import numpy as np
import os

def load_datasets():
    
    X=[]
    y=[]
    for image_label in label:
        images = os.listdir("images/train/"+image_label)
        for image in images:
            img = misc.imread("images/train/"+image_label+"/"+image)
            img = misc.imresize(img, (64, 64))
            X.append(img)
            y.append(label.index(image_label))
 
    X=np.array(X)
    y=np.array(y)
    return X,y

label = os.listdir("dataset_image")
save_label = open("int_to_word_out.pickle","wb")
pickle.dump(label, save_label)
save_label.close()