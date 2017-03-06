from __future__ import absolute_import
from __future__ import print_function
import time
import urllib, cStringIO
import progressbar as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Lambda, Flatten, Dense
from keras.regularizers import l2
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import RMSprop
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
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def vgg16():
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,224,224,))))
    seq.add(Flatten())
    return seq
    

def create_base_network():
    seq = Sequential()
    seq.add(Dense(128, activation='relu', W_regularizer=l2(1.25e-4), input_dim=25088)) 
    return seq


def load_and_preprocess(csv_file, root_url='http://ecx.images-amazon.com/images/I/'):
    data = pd.read_csv(csv_file, sep=';')
    k = 1
    
    for i in ['pic1', 'pic2']:
        print("Preprocessing part {} of 2.".format(k))
        widgets = ['Progress: ', pb.Percentage(), ' ', pb.Bar(marker='0',left='[',right=']'),
                   ' ', pb.ETA(), ' ', pb.FileTransferSpeed(), ' ']
        pbar = pb.ProgressBar(widgets=widgets, maxval=len(data))
        pbar.start()
                
        for j in range(len(data)):
            url = root_url + data[i][j]
            file = cStringIO.StringIO(urllib.urlopen(url).read())
            image = load_img(file, target_size=(224, 224))
            file.close()
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            data[i][j] = vgg16().predict(image)
            pbar.update(j)
        pbar.finish()
        time.sleep(1)
        print("Done.")
        time.sleep(1)
        k += 1
    
    print("All data preprocessed.")
    return data


def feature_scaling(data):
    max_value = np.max(data)
    data /= max_value
    return data
    

def create_pairdata():
    pairdata = []
    data = load_and_preprocess('data_mini.csv') 
    for i in range(len(data)):
        pic_1 = data['pic1'][i][0]
        pic_2 = data['pic2'][i][0]
        pairdata += [[pic_1, pic_2]]
    labels = data['score']
    return np.asarray(pairdata), np.asarray(labels)
   

def split_pairdata():
    pairdata, labels = create_pairdata()
    pairdata = feature_scaling(pairdata)
    print('Splitting data to get test set..') 
    X, X_test, y, y_test = train_test_split(pairdata, labels, 
                                                test_size=.2,
                                                random_state=7,
                                                stratify=labels)
    time.sleep(2)
    print('Splitting data to get validation set..')
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=.25,
                                                      random_state=7,
                                                      stratify=y)
                                                      
    return X_train, X_val, X_test, y_train, y_val, y_test  
    
    
def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

   
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
    return model
    

def train_and_predict(build_new=True):
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_pairdata()
    
    pos_pairs = np.concatenate((X_train[y_train==1], X_val[y_val==1], X_test[y_test==1]))
    neg_pairs = np.concatenate((X_train[y_train==0], X_val[y_val==0], X_test[y_test==0]))
    
    if build_new:
        model = siam_cnn()
        optimizer = RMSprop()
        model.compile(loss=contrastive_loss, optimizer=optimizer)
        print("Model compiled.")
    else:
        model = load_model('models/modelxx.h5', custom_objects={'contrastive_loss': contrastive_loss})
        print('Model loaded.')
        
    untrained_pred_pos = model.predict([pos_pairs[:,0], pos_pairs[:,1]])
    untrained_pred_neg = model.predict([neg_pairs[:,0], neg_pairs[:,1]])
    
    plt.figure(figsize=(4,4))
    plt.xlabel('Distance')
    plt.ylabel('Frequency')  
    sns.kdeplot(untrained_pred_neg[:,0], shade=True, color='red', label='Distant pairs')
    sns.kdeplot(untrained_pred_pos[:,0], shade=True, color='green', label='Close pairs')
    plt.legend(loc=1)
    #plt.savefig('untrained_pred.png')
    plt.show()
        
    model.fit([X_train[:,0], X_train[:,1]], y_train,
              validation_data = ([X_val[:,0], X_val[:,1]], y_val),
              batch_size=128,
              nb_epoch=10)
              
    time.sleep(5)
    print('Saving model..')    
    #model.save('models/modelxx.h5')
    print('Model saved.')
    
    trained_pred_pos = model.predict([pos_pairs[:,0], pos_pairs[:,1]])
    trained_pred_neg = model.predict([neg_pairs[:,0], neg_pairs[:,1]])
    
    plt.figure(figsize=(4,4))   
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    sns.kdeplot(trained_pred_neg[:,0], shade=True, color='red', label='Distant pairs')
    sns.kdeplot(trained_pred_pos[:,0], shade=True, color='green', label='Close pairs')
    plt.legend(loc=1)
    #plt.savefig('trained_pred.png')    
    plt.show()
   
    y_pred = model.predict([X_test[:,0], X_test[:,1]])
    return y_test, y_pred
    

def evaluate():
    # compute final accuracy on training and test sets
    y_test, y_pred = train_and_predict()
    te_acc = compute_accuracy(y_pred, y_test)
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    y_pred = 1 - y_pred 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    # Plot of a ROC curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label='ROC (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig('performance_data/ROC.png')
    plt.show()
    

if __name__ == '__main__':
    evaluate()