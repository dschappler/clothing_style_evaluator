import numpy as np
import pandas as pd
import cStringIO, urllib
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from siamese_net import contrastive_loss, create_base_network, create_pairdata
from sklearn.manifold import TSNE
from skimage.transform import resize


def prepare_model_and_data():
    trained_model = load_model('models/best_model.h5', 
                               custom_objects={'contrastive_loss': contrastive_loss})
    
    top_model = create_base_network()
    weights_layer = np.asarray([trained_model.layers[2].get_weights()[0], trained_model.layers[2].get_weights()[1]])
    top_model.layers[0].set_weights(weights_layer)

    pairdata, _ = create_pairdata()
    max_value = np.load('data/max_value.npy')
    pairdata /= max_value
    imgstream = pairdata[:,0]
    pred = top_model.predict(imgstream)
    return pred


def prepare_images():
    data = pd.read_csv('data/data_mini.csv', sep=";")['pic1']
    root_url = 'http://ecx.images-amazon.com/images/I/'
    for i in range(len(data)): 
        data[i] = root_url + data[i]
        file = cStringIO.StringIO(urllib.urlopen(data[i]).read())
        file = Image.open(file)
        data[i] = np.array(file, dtype=np.float)/255
        file.close()
    return data
    

def tsne():
    pred = prepare_model_and_data()
    tsne = TSNE()
    tsne_transformed = tsne.fit_transform(pred)
    plt.figure()
    plt.scatter(x=tsne_transformed[:,0], y=tsne_transformed[:,1])
    plt.show()
    return tsne_transformed

    
def min_resize(img, size):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    w, h = map(float, img.shape[:2])
    if min([w, h]) != size:
        if w <= h:
            img = resize(img, (int(round((h/w)*size)), int(size)))
        else:
            img = resize(img, (int(size), int(round((w/h)*size))))
    return img
    
    
def image_scatter(img_res=150, res=8000, cval=1.):
    """
    Embeds images via tsne into a scatter plot.

    Parameters
    ---------
    features: numpy array
        Features to visualize

    images: list or numpy array
        Corresponding images to features. Expects float images from (0,1).

    img_res: float or int
        Resolution to embed images at

    res: float or int
        Size of embedding image in pixels

    cval: float or numpy array
        Background color value

    Returns
    ------
    canvas: numpy array
        Image of visualization
    """
    images = prepare_images()
    images = [min_resize(image, img_res) for image in images]
    max_width = max([image.shape[0] for image in images])
    max_height = max([image.shape[1] for image in images])

    f2d = tsne()

    xx = f2d[:, 0]
    yy = f2d[:, 1]
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    # Fix the ratios
    sx = (x_max-x_min)
    sy = (y_max-y_min)
    if sx > sy:
        res_x = sx/float(sy)*res
        res_y = res
    else:
        res_x = res
        res_y = sy/float(sx)*res

    canvas = np.ones((res_x+max_width, res_y+max_height, 3))*cval
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    for x, y, image in zip(xx, yy, images):
        w, h = image.shape[:2]
        x_idx = np.argmin((x - x_coords)**2)
        y_idx = np.argmin((y - y_coords)**2)
        canvas[x_idx:x_idx+w, y_idx:y_idx+h] = image
   
    plt.figure(figsize=(60,60))
    plt.imshow(canvas)
    #plt.savefig('tsne.png')
    plt.show()
    
    
if __name__=='__main__':
    image_scatter()