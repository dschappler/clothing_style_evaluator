import urllib, cStringIO
from PIL import Image
import csv
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

def load_and_preprocess(URL):
    file = cStringIO.StringIO(urllib.urlopen(URL).read())
    image = load_img(file, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def pred_and_label(model, URL): 
    image = load_and_preprocess(URL)
    pred = model.predict(image)
    pred_label = (decode_predictions(pred)[0][0][1], decode_predictions(pred)[0][0][2])
    return pred_label 
    print('Predicted:', pred_label)
    
def show_img(file):
    img = Image.open(file)
    img.show()
    
if __name__ == "__main__":
    pred_and_label(ResNet50(weights='imagenet'), 'http://ecx.images-amazon.com/images/I/413epYRW4bL._SX342_.jpg')
