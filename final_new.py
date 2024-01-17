# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:47:45 2021

@author: Alexa

final project code
"""
import numpy as np 
import cv2
import glob
import os
import pandas as pd 
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
dimension = 100


#Create a function to pull out the fruit images from the folder 
#define function and give it a name
#has 2 inputs, the fruit you want and the data file you want it from (test/training)
#for example, if you want to load in pineapples in the training set and print 
#out the number of images: getFruitImages('pineapple', 'Training', print_n=True, k_fold=False)
#more context after the function we are defining 
def getFruitImages(fruit, data_type, print_n =False, k_fold=False):
        #create blank array to use
    images = []
    labels = []
    values = ['Training', 'Test']
    
#if statement that separates out the the true/false input into the function (k_fold)
#ths runs is k_fold=False
    if not k_fold:
        #define where the pictures are located
        path = "fruits-360/" + data_type + "/"
        #create for loop to read and print (if true) fruits
        #enumerate if just another way to define a loop that setting it to a number
        for i,j in enumerate(fruit):
            #this sets the location for the image
            p = path + j
            #this will be used for a counter later
            k = 0
            #create for loop for preccessing the images 
            for image_path in glob.glob(os.path.join(p, "*.jpg")):
                #read the color
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                #re-size the image
                image = cv2.resize(image, (dimension, dimension))
                #convert the color from rgb to bgr
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #add information to images and labels array
                images.append(image)
                labels.append(i)
                k+=1
            if(print_n):
                print(fruit[i].upper(), ": ", k, " ", data_type.upper())
        images = np.array(images)
        labels = np.array(labels)
    #return the information gained above to the user (see after functions)
        return images, labels
#if statement that separates out the the true/false input into the function (k_fold)
#This runs if k_fold = True
    else: 
        for v in values: 
             #define where the pictures are located
             path = "fruits-360/" + data_type + "/"
             for i,j in enumerate(fruit):
            #this sets the location for the image
                p = path + j
            #this will be used for a counter later
                k = 0
            #create for loop for preccessing the images 
                for image_path in glob.glob(os.path.join(p, "*.jpg")):
                #read the color
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    #re-size the image
                    image = cv2.resize(image, (dimension, dimension))
                    #convert the color from rgb to bgr
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    #add information to images and labels array
                    images.append(image)
                    labels.append(i)
                    k+=1
        images = np.array(images)
        labels = np.array(labels)
        return images, labels


fruits = ['Apple Braeburn', 'Pineapple']

#load in the data, this is done by calling the functions we did above
X_train, y_train =  getFruitImages(fruits, 'Training', print_n=True, k_fold=False)

#trainsform X_train to something that can be used to graph. This takes the array with color values and scales them
scaler = StandardScaler()
X_train2 = scaler.fit_transform([i.flatten() for i in X_train])
#working on ploting the images in 2D. It does not work completely yet
print('scaler done')
mds = MDS(n_components=2)
X_2D = mds.fit_transform(X_train2)
print(len(X_2D))
#creating the DataFrame 
df1 = pd.DataFrame({'Name': y_train}) 
df2 = pd.DataFrame({'x':X_2D[:,0], 'y': X_2D[:,1]})
result = pd.concat([df1, df2], axis=1, join='inner') 
# displaying the DataFrame 
print('DataFrame:\n', result)   
 # saving the DataFrame as a CSV file 
fruitdata = result.to_csv('fruit_data_MDS.csv', index = True) 
print('\nCSV String:\n', fruitdata) 
print('mds done')

# tsne = TSNE(n_components=2)
# X_2DTSNE = tsne.fit_transform(X_train2)
# #creating the DataFrame 
# df1tsne = pd.DataFrame({'Name': y_train}) 
# df2tsne = pd.DataFrame({'x':X_2DTSNE[:,0], 'y': X_2DTSNE[:,1]})
# resulttsne = pd.concat([df1tsne, df2tsne], axis=1, join='inner') 
# # displaying the DataFrame 
# print('DataFrame:\n', resulttsne)   
# # saving the DataFrame as a CSV file 
# fruitdatatsne = resulttsne.to_csv('fruit_data_TSNE.csv', index = True) 
# print('\nCSV String:\n', fruitdatatsne) 
# print('tsne done')

# pca = PCA(n_components=2)
# X_2Dpca = pca.fit_transform(X_train2)

# #creating the DataFrame 

# df1pca = pd.DataFrame({'Name': y_train}) 
# df2pca = pd.DataFrame({'x':X_2Dpca[:,0], 'y': X_2Dpca[:,1]})
# resultpca = pd.concat([df1pca, df2pca], axis=1, join='inner') 
# # displaying the DataFrame 
# print('DataFrame:\n', resultpca) 
   
#  # saving the DataFrame as a CSV file 
# fruitdatapca = resultpca.to_csv('fruit_data_pca.csv', index = True) 
# print('\nCSV String:\n', fruitdatapca) 
# print('pca done')