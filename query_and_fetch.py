"""
@author: dschappler
"""

import numpy as np
from keras.models import load_model, Sequential
from siamese_net import contrastive_loss, create_bottleneck_network, create_base_network, load_and_preprocess


def dist(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

trained_model = load_model('models/model6.h5', 
                           custom_objects={'contrastive_loss': contrastive_loss})


#TODO: load bottleneck features and extract single image-stream
    #pairs = np.load('bottleneck_data/bottleneck_pairs.npy')
    #imgstream = np.concatenate((pairs[:2000,0], pairs[:2000,1]), axis=0)
    #np.save('imgstream', imgstream)
imgstream = np.load('imgstream.npy')


#TODO: extract only one base network of model7 and run image-stream through it (gives 64dim-representations of each image)
top_model = create_base_network()

weights_layer_1 = np.asarray([trained_model.layers[2].get_weights()[0], trained_model.layers[2].get_weights()[1]])
top_model.layers[0].set_weights(weights_layer_1)

weights_layer_2 = np.asarray([trained_model.layers[2].get_weights()[2], trained_model.layers[2].get_weights()[3]])
top_model.layers[2].set_weights(weights_layer_2)
######################################

seq = Sequential()
seq.add(top_model)
seq.summary()
stylespace = seq.predict(imgstream)

bottom_model = create_bottleneck_network()

query = load_and_preprocess(URL)
query = bottom_model.predict(query)
pred = seq.predict(query)
np.sum(pred)

ret = []
for i in range(len(stylespace)):
    ret.append(dist(pred, stylespace[i]))
ret=np.asarray(ret)
ret=ret[ret>0]
best_indices_rotrock = np.argsort(ret)[:10]
worst_indices_rotrock = np.argsort(ret)[-10:]
#TODO: fetch nearest neighbour(s)

np.save('bestrock', best_indices_rotrock)
np.save('worstrock', worst_indices_rotrock)
np.save('besthose', best_indices_anzughose)
np.save('worsthose', worst_indices_anzughose)

#TODO: visualize NN
URL = 'https://mosaic01.ztat.net/vgs/media/pdp-gallery/N3/22/1B/00/QG/11/N3221B00Q-G11@10.jpg' ##rotrock
##URL='https://mosaic02.ztat.net/vgs/media/pdp-gallery/R1/92/2A/00/6C/11/R1922A006-C11@12.jpg' anzughose grau
file = cStringIO.StringIO(urllib.urlopen(URL).read())
img = Image.open(file)
img.show()
img.close()
#TODO: t-SNE
