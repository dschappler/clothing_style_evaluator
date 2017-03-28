# Clothing Style Evaluator
TODO: Write a project description

### Installation / Requirements

1. This project requires **Python 2.7** and the following Python packages installed:

- [keras] on [Theano]
- [matplotlib]
- [NumPy]
- [pandas]
- [Pillow]
- [progressbar]
- [seaborn]
- [scikit-image]
- [scikit-learn] (0.17.x)


2. Download the fully trained model by using the URL given in ```models/model_url.txt``` or clicking [this link](https://docs.google.com/uc?export=download&confirm=WXIN&id=0B2CX6USTeTN7clJHdkMzYjhfaU0). Make sure to save it as ```models/best_model.h5```.

3. If not already your current setting, change the precision that `keras` uses by setting `"floatx": "float64"` in  `~/.keras/keras.json`.


### Usage

The Style Evaluator works with both locally stored product images and image-URLs. 
Just run `python style_evaluator.py` in your command window or execute the first code cell of `Clothing_Style_Evaluator.ipynb` and follow the instructions.

For example, if we want to use a locally stored image of a jacket,

<img src="https://raw.githubusercontent.com/dschappler/clothing_style_evaluator/master/example_images/matching_1.jpg" height="250" />

we answer `local` to the first question and enter the local image path as an answer to the second question.


```Image 1 of 2:
Do you want to use a local image or an image-URL? Enter "local" for a local image or "url" for an image-URL: local
Please enter the local image path: C:\Users\dschappler\Pictures\matching_1.jpg
Loading your image...
Image successfully loaded.
```

We want to evaluate the combination of our jacket and some dress pants we found in an online store,

<img src="https://raw.githubusercontent.com/dschappler/clothing_style_evaluator/master/example_images/matching_2.jpg" height="250" />

so we answer `url` this time to the first question and enter the image-URL as an answer to the second question.

```Image 2 of 2:
Do you want to use a local image or an image-URL? Enter "local" for a local image or "url" for an image-URL: url
Please enter the image-URL: https://raw.githubusercontent.com/dschappler/clothing_style_evaluator/master/example_images/matching_2.jpg
Loading your image...
Image successfully loaded.
```

After the images are loaded, the stylistic similarity of the image pair gets evaluated:

```
Calculating the score...
This outfit gets a score of 0.57.
Your result:
Nice, this looks good together!
```

### Scripts

* ```siamese_net.py```: Trains and evaluates a siamese convolutional network on pairs of amazon.com clothing and jewelry product images with the aim to learn stylistic visual similarity.

* ```style_evaluator.py```: Application that evaluates the stylistic visual similarity of a pair of user input images of clothes or jewelry. Uses the learned weights from training the siamese network. Also the main input for ```Clothing_Style_Evaluator.ipynb```.

* ```tsne_vis.py```: Plotting a 2D-embedding of the 128D-stylespace that is the output of our siamese net.


### Report

For more insight and further information about this project and the surrounding problem domain, a report is available as ```report.pdf```.

### Dataset

We use a processed subset of the data set used in [Veit et al.] (available at https://goo.gl/blgCC2), which itself uses the large scale data set provided from [McAuley et al.]. The latter consists of product images from amazon.com, their respective product categories and product co-purchase information. A large majority of the images are iconic and have a white background, only some products are shown in a full-body picture.

Veit et al. only focus on the Clothing, Shoes and Jewelry category and its subcategories. As they are explicitly interested in cross-category fit, they use a strategic method to sample training data where pairs of data are so-called "heterogeneous dyadic co-occurences" (the two elements of a pair belong to different high-level categories and frequently co-occur).


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   
[NumPy]: <http://www.numpy.org>
[pandas]: <http://pandas.pydata.org/>
[keras]: <https://keras.io/>  
[Theano]: <http://deeplearning.net/software/theano/> 
[matplotlib]: <http://matplotlib.org/> 
[Pillow]: <https://python-pillow.org/>
[progressbar]: <https://pypi.python.org/pypi/progressbar>
[seaborn]: <https://seaborn.pydata.org/>
[scikit-image]: <http://scikit-image.org/>
[scikit-learn]: <http://scikit-learn.org>
[Veit et al.]: <https://arxiv.org/pdf/1509.07473v1.pdf>
[McAuley et al.]: <http://jmcauley.ucsd.edu/data/amazon/>
