'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
'''
from __future__ import absolute_import
from __future__ import print_function
import time
import urllib, cStringIO
import progressbar as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Flatten, Dense, Dropout
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import RMSprop#, SGD
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cross_validation import train_test_split


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
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))


def load_and_preprocess(URL):
    file = cStringIO.StringIO(urllib.urlopen(URL).read())
    image = load_img(file, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    file.close()
    return image
    

def load_images(csv_file):
    data = pd.read_csv(csv_file, sep=';')
    print('csv read.')
    root_url = 'http://ecx.images-amazon.com/images/I/'
    widgets = ['Progress: ', pb.Percentage(), ' ', pb.Bar(marker='0',left='[',right=']'),
           ' ', pb.ETA(), ' ', pb.FileTransferSpeed(), ' ']
    pbar = pb.ProgressBar(widgets=widgets, maxval=len(data))
    pbar.start()
    for i in range(len(data)): 
        data['pic1'][i] = load_and_preprocess(root_url + data['pic1'][i])
        data['pic2'][i] = load_and_preprocess(root_url + data['pic2'][i])
        pbar.update(i)
    pbar.finish()
    print('Images preprocessed.')           
    return data


def create_bottleneck_network():
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,224,224,))))
    seq.add(Flatten())
    return seq
    
    
def create_base_network():
    #TODO: add more regularization
    seq = Sequential()
    seq.add(Dense(64, activation='relu', input_dim=25088))
    seq.add(Dropout(0.1))
    seq.add(Dense(10, activation='relu'))
    return seq
    
    
def bottleneck_features(save=False):
    if save:
        data = load_images('train.csv')    
        print('Images loaded.')    
        model = create_bottleneck_network()
        print('Model loaded.')
        pairs = []
        widgets = ['Progress: ', pb.Percentage(), ' ', pb.Bar(marker='0',left='[',right=']'),
               ' ', pb.ETA(), ' ', pb.FileTransferSpeed(), ' ']
        pbar = pb.ProgressBar(widgets=widgets, maxval=len(data))
        pbar.start()
        for i in range(len(data)):
            pic_1 = model.predict(data['pic1'][i])[0]
            pic_2 = model.predict(data['pic2'][i])[0]
            pairs += [[pic_1, pic_2]]
            pbar.update(i)
        pbar.finish()
        print('Finished. Saving features...')
        np.save('bottleneck_pairs', np.asarray(pairs))
        np.save('bottleneck_labels', np.asarray(data['score']))
        print('Features saved.')
    
    print("Loading pairs..")
    pairdata = np.load('bottleneck_pairs.npy')
    time.sleep(3)
    print("Loading labels..")
    labeldata = np.load('bottleneck_labels.npy')
    time.sleep(3)
    print("Data loaded.")
    max_value = np.max(pairdata)
    pairdata /= max_value #(pairdata - np.min(pairdata)) / (np.max(pairdata) - np.min(pairdata))    
    print('Splitting data to get test set..')    
    X, X_test, y, y_test = train_test_split(pairdata, labeldata, 
                                            test_size=.2,
                                            random_state=7,
                                            stratify=labeldata)
    time.sleep(3)
    print('Splitting data to get validation set..')    
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=.25,
                                                      random_state=7,
                                                      stratify=y)
    del X, y
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.mean(labels == (predictions.ravel() < 0.5))
   
   
def siam_cnn():
    base_network = create_base_network()
    input_a = Input(shape=(25088,))
    input_b = Input(shape=(25088,))
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)
    #TODO: model.summary(), model.get_config(), model.get_weights(), model.plot()
    return model


def train_and_predict():
    X_train, X_val, X_test, y_train, y_val, y_test = bottleneck_features()
    model = siam_cnn()
    optimizer = RMSprop(clipnorm=0.1)
    model.compile(loss=contrastive_loss, optimizer=optimizer)
    print("Model compiled.")
    model.fit([X_train[:,0], X_train[:,1]], y_train,
              validation_data = ([X_val[:,0], X_val[:,1]], y_val),
              batch_size=128,
              nb_epoch=5)
    #TODO: train acc, learning curve, optimal stopping, save weights, save model HISTORY
    y_pred = model.predict([X_test[:,0], X_test[:,0]])
    return y_test, y_pred


def evaluate():
    # compute final accuracy on training and test sets
    y_test, y_pred = train_and_predict()
    te_acc = compute_accuracy(y_pred, y_test)
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(fpr, tpr)
       
    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    

#TODO: t-sne & visualization


if __name__=="__main__":
    evaluate()