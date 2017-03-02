# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 12:15:08 2017

@author: dschappler
"""
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

model = load_model('models/model9.h5', custom_objects={'contrastive_loss': contrastive_loss})
max_value = np.load('max_value.npy')

def local_or_url():
    mode = raw_input('local or url? : ')
    if mode == 'local':
        path = raw_input('enter local image path: ')
        try:
            img = load_img(path, target_size=(224, 224))
        except:
            print('not a valid image path or image file.')
            sys.exit(1)

    if mode == 'url':
        url = raw_input('enter image url: ')
        try:
            img = cStringIO.StringIO(urllib.urlopen(url).read())
            img = load_img(img, target_size=(224, 224))
        except:
            print('not a valid url or image.')
            sys.exit(1)
    
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
    
    
def evaluate_outfit():
    img_1 = push_through_vgg() / max_value
    time.sleep(3)
    img_2 = push_through_vgg() / max_value
    time.sleep(3)
    pred = model.predict([img_1, img_2])
    
    print('This outfit gets a score of {:1.2f}'.format(float(pred)))
    if float(pred)<.6:
        print('Nice, this looks good together!')
    elif .6 <= float(pred) <= .75:
        print('Hmm.. not sure..')
    else:
        print('Sorry, you better not wear this together.')


if __name__=="__main__":
    evaluate_outfit()