"""
@author: dschappler
"""

import time
import numpy as np
import pandas as pd
import cStringIO, urllib
from PIL import Image
from keras.models import load_model, Sequential
from siamese_net import contrastive_loss, create_bottleneck_network, create_base_network, load_and_preprocess, load_images


def dist(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

trained_model = load_model('models/model19.h5', 
                           custom_objects={'contrastive_loss': contrastive_loss})



#TODO: load bottleneck features and extract single image-stream

def load_data(pairdata=False):
    if pairdata:
        pairdata = np.load('bottleneck_data/bottleneck_pairs.npy')
        time.sleep(3)
        max_value = np.max(pairdata)
        pairdata /= max_value
        imgstream = pairdata[:17500,0]
        np.save('imgstream', imgstream)
    else:               
        imgstream = np.load('imgstream.npy')
        pairdata=None
        max_value = np.load('max_value.npy')
    return pairdata, imgstream, max_value

#TODO: extract only one base network of model7 and run image-stream through it (gives 64dim-representations of each image)
top_model = create_base_network()
bottom_model = create_bottleneck_network()

weights_layer_1 = np.asarray([trained_model.layers[2].get_weights()[0], trained_model.layers[2].get_weights()[1]])
top_model.layers[0].set_weights(weights_layer_1)


######################################

seq = Sequential()
seq.add(top_model)
seq.summary()

_, imgstream, max_value = load_data()
stylespace = seq.predict(imgstream)


query = load_and_preprocess('https://mosaic02.ztat.net/vgs/media/pdp-gallery/R1/92/2A/00/6C/11/R1922A006-C11@12.jpg')
query = bottom_model.predict(query/max_value)
pred = seq.predict(query)


#TODO: fetch nearest neighbour(s)
ret = []
for i in range(len(stylespace)):
    ret.append(dist(pred, stylespace[i]))
ret=np.asarray(ret)

best_indices_anzughose = np.argsort(ret)[:10]
worst_indices_anzughose = np.argsort(ret)[-10:]

############
np.save('bestrock', best_indices_rotrock)
np.save('worstrock', worst_indices_rotrock)
np.save('besthose', best_indices_anzughose)
np.save('worsthose', worst_indices_anzughose)


URL = 'https://mosaic01.ztat.net/vgs/media/pdp-gallery/N3/22/1B/00/QG/11/N3221B00Q-G11@10.jpg' ##rotrock
##URL='https://mosaic02.ztat.net/vgs/media/pdp-gallery/R1/92/2A/00/6C/11/R1922A006-C11@12.jpg' anzughose grau


#TODO: visualize NN
URL_rawdata = pd.read_csv('data.csv', sep=';')
URL_data = URL_rawdata['pic1'][:17500]   
root_url = 'http://ecx.images-amazon.com/images/I/'
for i in range(len(URL_data)): 
    URL_data[i] = root_url + URL_data[i]
    
for i in range(len(worst_indices_anzughose)):
    best_url = URL_data[worst_indices_anzughose[i]]
    file = cStringIO.StringIO(urllib.urlopen(best_url).read())
    img = Image.open(file)
    img.show()
    img.close()
    
#TODO: t-SNE

