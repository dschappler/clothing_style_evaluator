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

text

<img src="https://raw.githubusercontent.com/dschappler/clothing_style_evaluator/master/example_images/matching_1.jpg" height="250" />

```Image 1 of 2:
Do you want to use a local image or an image-URL? Enter "local" for a local image or "url" for an image-URL: local
Please enter the local image path: C:\Users\dschappler\Pictures\matching_1.jpg
Loading your image...
Image successfully loaded.
```

text

<img src="https://raw.githubusercontent.com/dschappler/clothing_style_evaluator/master/example_images/matching_2.jpg" height="250" />

```Image 2 of 2:
Do you want to use a local image or an image-URL? Enter "local" for a local image or "url" for an image-URL: url
Please enter the image-URL: https://raw.githubusercontent.com/dschappler/clothing_style_evaluator/master/example_images/matching_2.jpg
Loading your image...
Image successfully loaded.
```

text

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
