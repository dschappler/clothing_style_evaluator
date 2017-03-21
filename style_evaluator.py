'''Application that evaluates the stylistic visual similarity of a pair of user
input images of clothes or jewelry. Uses the learned weights from training the 
siamese network.
'''

from __future__ import absolute_import
from __future__ import print_function
import sys
import time
import urllib, cStringIO
import numpy as np
from siamese_net import contrastive_loss
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Sequential, load_model
from keras.layers import Input, Flatten


#Load trained model and value for rescaling
model = load_model('models/best_model.h5', custom_objects={'contrastive_loss': contrastive_loss})
max_value = np.load('data/max_value.npy')


def local_or_url():
    '''Lets a user choose between loading an image from a local drive or from 
    the internet. Preprocesses the image to be an input to VGG16.
    '''
    mode = raw_input('Do you want to use a local image or an image-URL? Enter "local" for a local image or "url" for an image-URL: ')
    if mode == 'local':
        path = raw_input('Please enter the local image path: ')
        try:
            img = load_img(path, target_size=(224, 224))
        except:
            print('This is not a valid image path or image file.')
            sys.exit(1)

    if mode == 'url':
        url = raw_input('Please enter the image-URL: ')
        try:
            img = cStringIO.StringIO(urllib.urlopen(url).read())
            img = load_img(img, target_size=(224, 224))
        except:
            print('This is not a valid url or image.')
            sys.exit(1)
    print('Loading your image...')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
   
   
def push_through_vgg():
    img = local_or_url()
    seq = Sequential()
    seq.add(VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,224,224,))))
    seq.add(Flatten())
    
    pred_img = seq.predict(img)
    print('Image successfully loaded.')
    return pred_img
    
    
def evaluate():
    '''Calculates the image difference in the latent style space and evaluates
    if the styles match and go well together.
    '''
    print('Image 1 of 2:')
    #rescaling
    img_1 = push_through_vgg() / max_value
    time.sleep(1)
    print('Image 2 of 2:')
    #rescaling
    img_2 = push_through_vgg() / max_value
    time.sleep(1)
    print('Calculating the score...')
    pred = model.predict([img_1, img_2])
    
    print('This outfit gets a score of {:1.2f}.'.format(float(pred)))
    print('Your result:')
    if float(pred)<=.82:
        print('Nice, this looks good together!')
    else:
        print('Sorry, these styles do not match.')


if __name__=="__main__":
    evaluate()