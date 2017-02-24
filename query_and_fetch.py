"""
@author: dschappler
"""

import time
import numpy as np
import pandas as pd
import cStringIO, urllib
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
from siamese_net import contrastive_loss, create_base_network
from sklearn.manifold import TSNE


trained_model = load_model('models/model9.h5', 
                           custom_objects={'contrastive_loss': contrastive_loss})


def load_data(pairdata=False):
    if pairdata:
        pairdata = np.load('bottleneck_data/bottleneck_pairs.npy')
        time.sleep(3)
        max_value = np.max(pairdata)
        pairdata /= max_value
        imgstream = pairdata[:1000,0]
        np.save('imgstream', imgstream)
    else:               
        imgstream = np.load('imgstream.npy')
        pairdata=None
        max_value = np.load('max_value.npy')
    return pairdata, imgstream, max_value


top_model = create_base_network()

weights_layer = np.asarray([trained_model.layers[2].get_weights()[0], trained_model.layers[2].get_weights()[1]])
top_model.layers[0].set_weights(weights_layer)


######################################

data = pd.read_csv('data.csv', sep=";")['pic1'][:1000]
root_url = 'http://ecx.images-amazon.com/images/I/'
for i in range(len(data)): 
    data[i] = root_url + data[i]
    file = cStringIO.StringIO(urllib.urlopen(data[i]).read())
    file = Image.open(file)
    data[i] = np.array(file, dtype=np.float)/255
    file.close()
np.save('tsne_images', data)


######################################

pred = top_model.predict(imgstream)

tsne = TSNE()
tsne_transformed = tsne.fit_transform(pred)
np.save('tsne_transformed', tsne_transformed)

plt.figure()
plt.scatter(x=tsne_transformed[:,0], y=tsne_transformed[:,1])
#plt.xlim(-200,200)
#plt.ylim(-200,200)
plt.show()
