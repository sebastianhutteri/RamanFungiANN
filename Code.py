import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os
import scipy as sp
from IPython.display import display
from ipywidgets import FloatProgress
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split
import seaborn as sns
import datetime

class Classifier:

    def __init__(self, samplefolder, wavenum, smoothness, asymmetry, nruns, layers, nodes, act, lr, opt, loss, epochs):
        self.samplefolder = samplefolder
        self.wavenum = wavenum
        self.smoothness = smoothness
        self.asymmetry = asymmetry
        self.nruns = nruns
        self.layers = layers
        self.nodes = nodes
        self.act = act
        self.lr = lr
        self.opt = opt
        self.loss = loss
        self.epochs = epochs

        #The measurements of the fungal species were split over several files.
        self.NSessions = [3, 3, 3, 3, 3, 3, 7, 7, 3, 3, 3, 7, 3, 3, 3, 3] #How many files each species has. Same order in list as the files are read by Python.
        self.NSpecies = len(self.NSessions) #Number of different species.
        NSpectra = [66, 66, 66, 66, 66, 66, 18, 18, 66, 66, 66, 18, 66, 66, 66, 66] #Number of spectra per individual species file.

        Identity = np.identity(self.NSpecies) #Generates a one hot matrix of unit vectors used as validation data.
        Y = []
        for i in range(self.NSpecies):
            for j in range(self.NSessions[i]):
                Z = []
                for k in range(NSpectra[i]):
                    Z.append(list(Identity[i]))
                Y.append(Z)
        self.Y = Y

        self.SpeciesNames = ['AGP', 'AUP', 'CAC', 'CON20',
                            'CON25', 'C', 'EnzC2_24h3', 'EnzC2_48h2',
                            'GYM', 'G', 'LEPSP', 'NaOH3M',
                            'OP', 'PHALA', 'PSS', 'TEN'] #Labels for heatmap.

    def ALSS(self, y, niter=10,): #ALSS normalization.
        L = len(y)
        D = sp.sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sp.sparse.spdiags(w, 0, L, L)
            Z = W + self.smoothness * D.dot(D.transpose())
            z = sp.sparse.linalg.spsolve(Z, w * y)
            w = self.asymmetry * (y > z) + (1 - self.asymmetry) * (y < z)
        return z

    def FitData(self, fileindex): #Quadratic interpolation.
        ALSSData = []
        Data = pd.read_csv('{}/{}'.format(self.samplefolder, fileindex), sep=';', header=None)
        for i in range(1, len(Data.columns)):
            f = interp1d(Data[0], Data[i], kind='quadratic')
            y1 = f(self.wavenum)
            y2 = self.ALSS(y1)
            y = y1 - y2
            ALSSData.append(y)
        return ALSSData

    def ProcessData(self): #Interpolates and normalizes data using functions above, then stores it in a matrix.
        fp = FloatProgress(min=0,max=len(os.listdir(self.samplefolder)))
        display(fp)
        DataMatrix = []
        print('Storing data in DataMatrix...')
        for i in sorted(os.listdir(self.samplefolder)):
            DataMatrix.append(self.FitData(i))
            fp.value += 1
        self.DataMatrix = DataMatrix
        print('Data stored.')

    def XTrain(self, index): #Functions for slicing the data matrix into training and validation data.
        return np.concatenate(self.DataMatrix[:index] + self.DataMatrix[index + 1:])
    def YTrain(self, index):
        return np.concatenate(self.Y[:index] + self.Y[index + 1:])
    def XTest(self, index):
        return np.array(self.DataMatrix[index])
    def YTest(self, index):
        return np.array(self.Y[index])

    def ANN(self, index): #Constructs the ANN.
        model = Sequential()
        model.add(Dense(self.nodes, activation=self.act, input_dim=len(self.wavenum))) #Input layer.

        for i in range(self.layers): #Adding selected number of hidden layers.
            model.add(Dense(self.nodes, activation=self.act))
        model.add(Dense(self.NSpecies, activation='softmax'))

        model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'])
        model.fit(self.XTrain(index), self.YTrain(index), epochs=self.epochs, batch_size=128) #Training the network.

        #Classification.
        predict = model.predict(self.XTest(index)) #Generates a prediction distrubution among classrs for every spectra.
        predictclass = np.argmax(predict, axis=1) #Returns the indices of the classes with the highest prediction.
        predictcount = np.zeros(self.NSpecies) #Array of zeros.
        for i in predictclass:
            predictcount[i] += 1 #Every max prediction adds 1 to its class.

        return predictcount/sum(predictcount) #Normalizes the classifier distribution.

    def SingleRun(self, run): #Single run of training and evaluating the network as well as generating a heatmap of the results.
        Pred = []
        for i in range(len(os.listdir(self.samplefolder))):
            Pred.append(self.ANN(i))
        Pred = np.array(Pred)
        Now = datetime.datetime.now()
        Stamp = ' {}{}{} {}.{}.{}'.format(Now.year, Now.month, Now.day, Now.hour, Now.minute, Now.second) #Time stamp.
        np.savetxt('{}/{}'.format(self.FolderStamp, 'Pred' + Stamp), Pred) #Saving data.

        Accuracy = []
        for i in range(self.NSpecies):
            Accuracy.append(sum(Pred[sum(self.NSessions[:i]):sum(self.NSessions[:i]) + self.NSessions[i]])/self.NSessions[i])
        Accuracy = np.array(Accuracy)
        AccuracyData = pd.DataFrame(Accuracy, columns=self.SpeciesNames, index=self.SpeciesNames) #Average accuracy for species.
        np.savetxt('{}/{}'.format(self.FolderStamp, 'Acc {}'.format(run)), Accuracy)
        plt.figure(figsize=(self.NSpecies,self.NSpecies))
        sns.heatmap(AccuracyData, annot=True)
        plt.savefig('{}/{}'.format(self.FolderStamp, 'Acc {}'.format(run) + '.png'))

        Perf = []
        for i in range(self.NSpecies):
            Perf.append(Accuracy[i][i])
        return sum(Perf)/self.NSpecies


    def Run(self): #Running the single run multiple times.
        Now = datetime.datetime.now()
        FolderStamp = 'Run {}{}{} {}.{}.{}'.format(Now.year, Now.month, Now.day, Now.hour, Now.minute, Now.second) #Time stamp.
        self.FolderStamp = FolderStamp
        os.makedirs(self.FolderStamp)

        parameters = 'WaveNumMin={}, WaveNumMax={}, WaveNumValues={}, ALSSSmoothness={}, ALSSAsymmetry={}, HiddenLayers={}, Nodes={}, Activation={}, LearningRate={}, Optimization={}, Loss={}, Epochs={}'.format(min(self.wavenum), max(self.wavenum), len(self.wavenum), self.smoothness, self.asymmetry, self.layers, self.nodes, self.act, self.lr, self.opt, self.loss, self.epochs)
        p = open(self.FolderStamp + '/Parameters.txt', 'w+')
        p.write(parameters)
        p.close()

        fp = FloatProgress(min=0,max=self.nruns)
        display(fp)
        print('Running network...')
        PerfList = []
        for i in range(self.nruns):
            PerfList.append(self.SingleRun(i + 1))
            fp.value += 1

        np.savetxt(self.FolderStamp + '/Performance', PerfList)
        return PerfList

#Parameters to bet set before running the code.
SampleFolder = 'Samples' #Folder name string of .csv-files.
WaveNum = np.linspace(1000, 1500, 500) #Numpy array of spectral data to be extracted.
Smoothness = 1e6 #Smoothness parameter of the ALSS.
Asymmetry = 0.001 #Asymmetry parameter of the ALSS.
NRuns = 5 #Number of training/evaluation iterations.
Layers = 4 #Number of hidden layers for the ANN.
Nodes = 32 #Number of nodes for the ANN.
Activation = 'relu' #Keras activation function string for input layer and hidden layers.
LearningRate = 1e-4 #Keras learning rate.
Optimization = Adam(lr=LearningRate) #Keras optimization algorithm string.
Loss = 'categorical_crossentropy' #Keras loss function string.
Epochs = 100 #Number of epochs.

#How to run code.
Pipeline = Classifier(SampleFolder, WaveNum, Smoothness, Asymmetry, NRuns, Layers, Nodes, Activation, LearningRate, Optimization, Loss, Epochs)
Pipeline.ProcessData()
Pipeline.Run()
