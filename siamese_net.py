'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Flatten, Dense
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import urllib, cStringIO
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import RMSprop

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def load_and_preprocess(URL):
    file = cStringIO.StringIO(urllib.urlopen(URL).read())
    image = load_img(file, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image
    
def load_images(csv_file):
    data = pd.read_csv(csv_file, sep=';')
    root_url = 'http://ecx.images-amazon.com/images/I/'
    labels = data['score']
    pic_1 = load_and_preprocess(root_url + data['pic1'][0])
    pic_2 = load_and_preprocess(root_url + data['pic2'][0])
    return np.array(pic_1, pic_2), np.array(labels)

def pred_and_label(model, URL): 
    image = load_and_preprocess(URL)
    pred = model.predict(image)
    pred_label = (decode_predictions(pred)[0][0][1], decode_predictions(pred)[0][0][2])
    return pred_label 
    print('Predicted:', pred_label)

def create_pairs():
    '''Positive and negative pair creation.
    '''
    pairs = []
    labels = []
    z1, z2 = load_and_preprocess('http://www.pixempire.com/images/preview/black-circle-icon.jpg'), load_and_preprocess('http://www.masadaband.net/wp-content/uploads/2014/09/black-square.jpg')
    pairs += [z1, z2]
    labels += [0]
    return np.array(pairs), np.array(labels)


def create_base_network():
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,224,224))))
    seq.add(Flatten())
    seq.add(Dense(256, activation='relu'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.mean(labels == (predictions.ravel() > 0.5))


# create training+test positive and negative pairs
tr_pairs, tr_y = create_pairs()
##tr_pairs, tr_y = load_images('val.csv')


# network definition
base_network = create_base_network()

input_a = Input(shape=(3,224,224))
input_b = Input(shape=(3,224,224))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)
#model.summary()
#model.get_config()
#model.get_weights()

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[0], tr_pairs[1]], tr_y,
          batch_size=1,
          nb_epoch=1)

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[0], tr_pairs[1]])
evalu = model.evaluate([tr_pairs[0], tr_pairs[1]], tr_y)
testu = model.test_on_batch([tr_pairs[0], tr_pairs[1]], tr_y)
tr_acc = compute_accuracy(pred, tr_y)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
