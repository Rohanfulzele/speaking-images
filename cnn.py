
#Building the cnn

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing the cnn

classifier=Sequential()

#Adding Convolution layer 
#In this step feature detectors(filters) are selected at random and applied the input image to extract feature maps
 
classifier.add(Conv2D(32 , (3,3) , input_shape=(64,64,3) , activation='relu' ))

#Adding MaxPooling layer to reduce the feature map size while still retaining the spatial information

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flatten the feature maps to convert the feature maps into 1 single row vector which would act as input neurons to future Neural network
#for eg. if we have 32 feature maps of size 31x31 coming out of pooling layer then it will be flattened to 32*31*31 inputs which are passed to next layer 

classifier.add(Flatten())

#Fully Connected layer

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=7, activation='softmax'))

#Compiling the cnn

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#fitting the cnn to image
#We also apply image augmentation to increase the size of our dataset by generating images by performing different transformations on them.

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)


training_set=train_datagen.flow_from_directory('data/training_set',
                                               target_size=(64,64),
                                               batch_size=32,# i.e 8000/32 ==256 batches
                                               class_mode='categorical')

test_set=test_datagen.flow_from_directory('data/test_set',
                                               target_size=(64,64),
                                               batch_size=32,
                                               class_mode='categorical')


#steps_per_epoch=It is the number of batches which is then used to calculate number of images in training set i.e (8000/32)*32==8000
#validation_steps=Used to calculate number of images in test test same as above

classifier.fit_generator(training_set,
                         steps_per_epoch=1560/32, 
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=200/32 ) 

#saving the weights of a model to use it further
classifier.save('my_model.h5')


from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

#image file names
#img1.jpg
#img2.jpg
#img3.jpg
#img4.jpeg
#img5.jpg
#img6.jpeg
#img7.jpg


img = imread('data/single_prediction/img1.jpg')
plt.axis("off")
plt.imshow(img,interpolation='nearest',aspect='auto')
plt.show()
img = resize(img,(64,64), mode='constant')

# reshape(batch_size(here only one image), height, width, channels)
#the neural networks excepts inputs in form of batch hence to need to add the 4th dimension which corresponds to batch size(here 1 as only 1 image)
img = np.reshape(img,(1,64,64,3))

#predicts the class for the input image 
#also can use classifier.predict(img) which gives the probability for the prediction
prediction = classifier.predict_classes(img)

classes=training_set.class_indices

import time
from playsound import playsound


def playSound(key):
    if(key==0):
        playsound('sounds/fire.mp3')
    elif(key==1):
        playsound('sounds/forest.mp3')
    elif(key==2):
        playsound('sounds/rain.mp3')
    elif(key==3):
        playsound('sounds/river.mp3')
    elif(key==4):
        playsound('sounds/thunder.mp3')
    elif(key==5):
        playsound('sounds/tornado.mp3')
    elif(key==6):
        playsound('sounds/waterfall.mp3')
    time.sleep(5)
    
for key,value in classes.items():
    if(value==prediction[0]):
        playSound(value)



#print(classifier.summary())