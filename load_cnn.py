from keras.models import load_model

classifier=load_model('my_model.h5')

from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

#image file names
#img1.jpg
#img2.jpg
#img3.jpg
#img4.jpg
#img5.jpg
#img6.jpg
#img7.jpg

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
            
            
def predict_n_play(i):
    
    img = imread('data/single_prediction/img'+str(i)+'.jpg')
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
    
    classes={'fire':0,
             'forest':1,
             'rain':2,
             'river':3,
             'thunder':4,
             'tornado':5,
             'waterfall':6
             } 
    
    for key,value in classes.items():
        if(value==prediction[0]):
            playSound(value)
            break
            
predict_n_play(13)

