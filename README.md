# Evolution Constructed Features / Genetic algorithm for image classification
This repo provides an image classification algorithm based on a genetic algorithm for feature optimization and a RandomForestClassifiers-based ensempble classifier (adaboost) for prediction.

The initial genetic algorithm for image classification has been described in Lillywhite, et al.: http://dx.doi.org/10.1016/j.patcog.2013.06.002. The repo closely follows the proposed algorithm with some additional image processing kernels. 

## Installation
The repo is managed with poetry. To install the environment run
```
    poetry install
```
from the root directory.

## Usage
To use the algorithm a library of images is required as well as a csv-file containing the labels and paths for each image. See `classification.csv` for more information. 

There are two ways of training the classification model.
- Interactive: Load `main.py` as module and create an instance of the `EvoFeatures`-class. Call `load_data` and provide a path to the directory where the csv-file is located. Afterwards call `fit` and provide and output-directory.
- Script: Modify the `if __name__`-section at the end of `main.py` and modify line number 875 to point to the directory with the classification-csv-file. Then run the entire script.

In both cases the training ends with the creation of an `adaboost.pkl` file in the output directory.

## Inference
- Load `main.py` as module and create an instance of the `EnsembleClassifier`-class, providing a path to the previously created adaboost-file.
- Load a set of images as a list of numpy arrays.
- Call the `predict`-method of the `EnsembleClassifier`-class and provide the list of images as argument

