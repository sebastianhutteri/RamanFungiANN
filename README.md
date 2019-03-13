# RamanFungiANN

This is the Python 3 code for my bachelor's project "Analysing Raman spectra of crystalline cellulose degradation by fungi using artificial neural networks".

The data pipeline does the following:
* Extracts data from .csv-files within a chosen folder
* Interpolates data using quadratic interpolation
* Normalizes data using "ALSS" or "Assymetric Least Squares Smoothing"
* Constructs and trains an artificial neural network classifier using Keras
* Evaluates performance and generates a heatmap of the classification results

The following parameters are to be set before running the code:
* SampleFolder (Folder name string)
* WaveNum (Numpy array of spectral data to be extracted)
* Smoothness (Smoothness parameter of the ALSS)
* Asymmetry (Asymmetry parameter of the ALSS)
* NRuns (Number of training/evaluation iterations)
* Layers (Number of layers for the ANN)
* Nodes (Number of nodes for the ANN)
* Activation (Keras activation function string for input layer and hidden layers)
* LearningRate (Keras learning rate string)
* Optimization (Keras optimization algorithm string)
* Loss (Keras loss function string)
* Epochs (Number of epochs)
