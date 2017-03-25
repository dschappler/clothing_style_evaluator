# Clothing Style Evaluator
TODO: Write a project description

### Installation / Requirements

This project requires **Python 2.7** and the following Python packages installed:

- [keras] on [Theano]
- [matplotlib]
- [NumPy]
- [pandas]
- [Pillow]
- [progressbar]
- [seaborn]
- [scikit-image]
- [scikit-learn] (0.17.x)


In addition to that, download the fully trained model by using the URL given in ```models/model_url.txt``` or clicking [this link](https://docs.google.com/uc?export=download&confirm=WXIN&id=0B2CX6USTeTN7clJHdkMzYjhfaU0). Make sure to save it as ```models/best_model.h5```.


### Usage
TODO: Write usage instructions

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


