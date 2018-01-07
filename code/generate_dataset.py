import cv2
import os
import keras
import numpy as np
import pickle as pk
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np



paths = 'C:/Users/Pakhi/Hackerrank - predict annual returns/ASL/asl_dataset'


dim = 50
classes = {
    '0': 0,     '1': 1,     '2': 2,     '3': 3,     '4': 4,     '5': 5,     '6': 6,     '7': 7,     '8': 8,     '9': 9,    
    'a': 10,    'b': 11,    'c': 12,    'd': 13,    'e': 14,    'f': 15,    'g': 16,    'h': 17,    'i': 18,    'j': 19,
    'k': 20,    'l': 21,    'm': 22,    'n': 23,    'o': 24,    'p': 25,    'q': 26,    'r': 27,    's': 28,    't': 29,    
    'u': 30,    'v': 31,    'w': 32,    'x': 33,    'y': 34,    'z': 35,}


dataset_file = "dataset.pickle"


def generate_training_dataset():
    
    print("Process begins!")
    
    image_set = []
    label_set = []
    for path in os.listdir(paths):

        image = []
        label = []

        for filename in os.listdir(paths+"/"+path):

            img_path = paths + "/" + path + "/" + filename
            img = cv2.imread(img_path) 
            
            

            img = cv2.resize(img,(dim, dim))        
            img = np.reshape(img,(-1)) 
            img = img.astype('float32')

            image.append(img)     
            label.append(classes[path]) 

        image_set.extend(image)
        label_set.extend(label)
        
        pk.dump([image_set,label_set],open(dataset_file,"wb"))
   
    print("Dataset file successfully created.")
        
    image_set, label_set =  shuffle(image_set, label_set)
    print("Dataset successfully shuffled.")
    
    return image_set, label_set
    

    

    
    

def shuffle(image_set, label_set):
    
    combined = list(zip(image_set, label_set))
    np.random.shuffle(combined)
    image_set[:], label_set[:] = zip(*combined)
    
    return image_set, label_set




def testtrain_split(image_set, label_set):
    
    x_train, x_test, y_train, y_test = train_test_split(image_set, label_set, 
                                                        test_size=0.2, 
                                                        random_state=42)    
    return x_train, x_test, y_train, y_test





def reshape_train_test(image_set, label_set, getShape = False):
    
    x_train, x_test, y_train, y_test = testtrain_split(image_set, label_set)
    
    #reshaping x_train
    X_train = np.asarray(x_train).astype('float32')
    X_train = np.reshape(x_train,(-1,dim,dim,1))
    X_train = X_train / 255.0
    
    #reshaping x_test
    X_test = np.asarray(x_test).astype('float32')
    X_test = np.reshape(x_test,(-1,dim,dim,1))
    X_test = X_test / 255.0
    
    #reshaping y_train
    Y_train = np.asarray(y_train)
    Y_test = np.asarray(y_test)
    
    #one_hot_encode y_train
    y_in = keras.utils.to_categorical(Y_train, 36)
    y_out = keras.utils.to_categorical(Y_test, 36)
    
    if(getShape == True):
        print("Dimensions of X_train:", X_train.shape)
        print("Dimensions of Y_train:", y_in.shape)
        print("Dimensions of X_test:", X_train.shape)
        print("Dimensions of Y_test:", y_out.shape)
        
    return X_train, X_test, y_in, y_out
        

#create_train_test_data(training_file,0.2)





def __main__():
       
    # create dataset
    generate_training_dataset()   

