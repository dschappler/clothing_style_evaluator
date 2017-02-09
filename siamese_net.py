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
import seaborn as sns
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Lambda, Flatten, Dense, Dropout
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
    seq = Sequential()
    seq.add(Dense(128, activation='relu', W_regularizer=l2(2e-4), input_dim=25088)) #
    #seq.add(Dropout(0.2))
    #seq.add(Dense(128, activation='relu')) #, W_regularizer=l2(1e-4)
    #seq.add(Dropout(0.2))    
    #seq.add(Dense(128, activation='relu'))
    return seq
    
    
def bottleneck_features(save=False, generate=False):
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
        np.save('bottleneck_data/bottleneck_pairs', np.asarray(pairs))
        np.save('bottleneck_data/bottleneck_labels', np.asarray(data['score']))
        print('Features saved.')
    
    if generate:
        print("Loading pairs..")
        pairdata = np.load('bottleneck_data/bottleneck_pairs.npy')
        max_value = np.max(pairdata)
        pairdata /= max_value #(pairdata - np.min(pairdata)) / (np.max(pairdata) - np.min(pairdata))
        time.sleep(3)    
        print("Loading labels..")
        labeldata = np.load('bottleneck_labels.npy')
        print("Data loaded.")
        time.sleep(3)    
        print('Splitting data to get test set..')    
    
        X, X_test, y, y_test = train_test_split(pairdata, labeldata, 
                                                test_size=.2,
                                                random_state=7,
                                                stratify=labeldata)
        time.sleep(3)
        pairdata, labeldata = None, None
        print('Splitting data to get validation set..')
        X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                          test_size=.25,
                                                          random_state=7,
                                                          stratify=y)
        X, y = None, None
        np.save('X_train', X_train) 
        np.save('X_val', X_val)
        np.save('X_test', X_test)
        np.save('y_train', y_train)
        np.save('y_val', y_val)
        np.save('y_test', y_test)
    
    print('Loading data..')    
    X_train = np.load('split_data/X_train.npy')     
    X_val = np.load('split_data/X_val.npy')
    X_test = np.load('split_data/X_test.npy')
    y_train = np.load('split_data/y_train.npy')
    y_val = np.load('split_data/y_val.npy')
    y_test = np.load('split_data/y_test.npy')
    print('Data loaded.')
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


def train_and_predict(build_new=False):
    
    X_train, X_val, X_test, y_train, y_val, y_test = bottleneck_features()
    
    if build_new:
        model = siam_cnn()
        optimizer = RMSprop() #clipnorm=0.1
        model.compile(loss=contrastive_loss, optimizer=optimizer)
        print("Model compiled.")
    else:
        model = load_model('models/model17.h5', custom_objects={'contrastive_loss': contrastive_loss})
        print('Model loaded.')
        
    model.fit([X_train[:,0], X_train[:,1]], y_train,
              validation_data = ([X_val[:,0], X_val[:,1]], y_val),
              batch_size=128,
              nb_epoch=1)
              
    time.sleep(5)
    print('Saving model..')    
    model.save('models/model17.h5')
    print('Model saved.')
    y_pred = model.predict([X_test[:,0], X_test[:,1]])
    return y_test, y_pred


def evaluate():
    # compute final accuracy on training and test sets
    y_test, y_pred = train_and_predict()
    te_acc = compute_accuracy(y_pred, y_test)
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
     ########
    np.save('performance_data/fpr_model17.npy', fpr)
    np.save('performance_data/tpr_model17.npy', tpr)
    np.save('performance_data/roc_auc_model17.npy', roc_auc)
    np.save('performance_data/y_pred_model17.npy', y_pred)
    
    
    #########
    fpr1=np.load('performance_data/fpr_model1.npy')
    tpr1=np.load('performance_data/tpr_model1.npy')
    roc_auc1=np.load('performance_data/roc_auc_model1.npy')
    
    fpr2=np.load('performance_data/fpr_model2.npy')
    tpr2=np.load('performance_data/tpr_model2.npy')
    roc_auc2=np.load('performance_data/roc_auc_model2.npy')
    
    fpr3=np.load('performance_data/fpr_model3.npy')
    tpr3=np.load('performance_data/tpr_model3.npy')
    roc_auc3=np.load('performance_data/roc_auc_model3.npy')
    
    fpr4=np.load('performance_data/fpr_model4.npy')
    tpr4=np.load('performance_data/tpr_model4.npy')
    roc_auc4=np.load('performance_data/roc_auc_model4.npy')
        
    fpr5=np.load('performance_data/fpr_model5.npy')
    tpr5=np.load('performance_data/tpr_model5.npy')
    roc_auc5=np.load('performance_data/roc_auc_model5.npy')
    
    fpr6=np.load('performance_data/fpr_model6.npy')
    tpr6=np.load('performance_data/tpr_model6.npy')
    roc_auc6=np.load('performance_data/roc_auc_model6.npy')
    
    fpr7=np.load('performance_data/fpr_model7.npy')
    tpr7=np.load('performance_data/tpr_model7.npy')
    roc_auc7=np.load('performance_data/roc_auc_model7.npy')
        
        
    
    # Plot of a ROC curve
    plt.figure(figsize=(9,9))
    #plt.plot(fpr1, tpr1, color='green', label='Model 1 (area = %0.3f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='blue', label='Model 2 (area = %0.3f)' % roc_auc2)
    plt.plot(fpr3, tpr3, color='red', label='Model 3 (area = %0.3f)' % roc_auc3)
    #plt.plot(fpr4, tpr4, color='black', label='ROC curve (area = %0.3f)' % roc_auc4)
    plt.plot(fpr5, tpr5, color='grey', label='Model 5 (area = %0.3f)' % roc_auc5)
    plt.plot(fpr6, tpr6, color='green', label='Model 6 (area = %0.3f)' % roc_auc6)
    plt.plot(fpr7, tpr7, color='black', label='Model 7 (area = %0.3f)' % roc_auc7)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    

def distplots():
    #labels = np.load('bottleneck_data/bottleneck_labels.npy')
    #pos_pairs = pairdata[labels==1]
    #neg_pairs = pairdata[labels==0]
    #np.save('pos_pairs', pos_pairs)
    #np.save('neg_pairs', neg_pairs)
    pos_pairs = np.load('pos_pairs.npy')
    neg_pairs = np.load('neg_pairs.npy')
    
    untrained_model = siam_cnn()
    trained_model = load_model('models/model7.h5', custom_objects={'contrastive_loss': contrastive_loss})
    
    untrained_pred_pos = untrained_model.predict([pos_pairs[:,0], pos_pairs[:,1]])
    untrained_pred_neg = untrained_model.predict([neg_pairs[:,0], neg_pairs[:,1]])
    trained_pred_pos = trained_model.predict([pos_pairs[:,0], pos_pairs[:,1]])
    trained_pred_neg = trained_model.predict([neg_pairs[:,0], neg_pairs[:,1]])
    
    plt.figure(figsize=(9,9))
    sns.kdeplot(untrained_pred_neg[:,0], shade=True, color='red')
    sns.kdeplot(untrained_pred_pos[:,0], shade=True, color='green')
    plt.show()
      
    plt.figure(figsize=(9,9))
    sns.kdeplot(trained_pred_neg[:,0], shade=True, color='red')
    sns.kdeplot(trained_pred_pos[:,0], shade=True, color='green')
    plt.show()


#TODO: t-sne & visualization

if __name__=="__main__":
    evaluate()