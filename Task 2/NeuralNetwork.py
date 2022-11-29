import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pre_processing import *


class NeuralNetwork():
    def __init__(self, SelectedFeatures, SelectedClasses, Learning_Rate, Num_epochs, Data, Bias , mse):
        self.Init(SelectedFeatures, SelectedClasses, Learning_Rate, Num_epochs, Data, Bias , mse)

    def plot_All_Data(self, Data, SelectedFeatures, SelectedClasses):
        self.XFeautresData, self.YData = Pre_Processing(Data.iloc[:, 1:], Data.iloc[:, 0:1], text='Plot')
        Unique_Y_Values = self.YData['species'].unique()
        self.XFeautresData = self.XFeautresData[SelectedFeatures]
        classe_1 = self.XFeautresData[self.YData['species'] == Unique_Y_Values[0]]
        classe_2 = self.XFeautresData[self.YData['species'] == Unique_Y_Values[1]]
        classe_3 = self.XFeautresData[self.YData['species'] == Unique_Y_Values[2]]
        plt.figure('Before train')
        plt.scatter(classe_1[SelectedFeatures[0]], classe_1[SelectedFeatures[1]], c='green')
        plt.scatter(classe_2[SelectedFeatures[0]], classe_2[SelectedFeatures[1]], c='red')
        plt.scatter(classe_3[SelectedFeatures[0]], classe_3[SelectedFeatures[1]], c='blue')
        plt.xlabel(SelectedFeatures[0])
        plt.ylabel(SelectedFeatures[1])
        plt.show()

    def Init(self, SelectedFeatures, SelectedClasses, Learning_Rate, Num_epochs, Data, Bias ,mse):
   #     self.plot_All_Data(Data, SelectedFeatures, SelectedClasses)
        self.mse = mse
        Data = Data[Data['species'].isin(SelectedClasses)]
        self.X = Data[SelectedFeatures]
        self.Y = Data['species']
        self.SelectedClasses = SelectedClasses
        self.X, self.Y = Pre_Processing(self.X, self.Y, SelectedClasses)
        self.bias = Bias
        self.Learning_Rate = Learning_Rate
        self.epochs = Num_epochs
        self.totalErrors = 0
        self.SelectedFeatures = SelectedFeatures
        if (Bias == 1):
            self.X.insert(0, 'Ones', 1)
            self.weights = np.random.rand(1, 3)
        else:
            self.weights = np.random.rand(1, 2)

        self.trainedX = self.X
        self.trainedY = self.Y
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.Split_Data()

    def Tarin(self):
        lastWeight = 0
        for i in range(self.epochs):
            self.totalErrors = 0
            for j in range(self.Xtrain.shape[0]):
                value = np.dot(self.weights[0].T, self.Xtrain[j])
                predicted = self.predictedFunction(value)
                Error = self.Error(predicted, self.YTrain[j])
                lastWeight = self.weights = self.weights - (self.Learning_Rate * Error * self.Xtrain[j])
                # self.totalErrors = self.totalErrors + np.abs(Error)
            total = 0
            for k in range(self.Xtrain.shape[0]):
                value = np.dot(lastWeight, self.Xtrain[k])
                error = (self.YTrain[k] - value)**2
                total = total + error
            MSE = (total/2)/self.Xtrain.shape[0]
            if (MSE < self.mse): break
        return self.predict()

    def predictedFunction(self, value):
        if (value >= 0):
            return 1
        return -1

    def predict(self):
        value = np.dot(self.Xtest, self.weights[0])

        predicted = []
        for i in range(value.shape[0]):
            predicted.append(self.predictedFunction(value[i]))

        # Listed = [1 if predicted[i] == self.Ytest[i] else 0 for i in range(self.Ytest.shape[0])]
        # Accuracy = Listed.count(1)/len(Listed)
        Accuracy = accuracy_score(self.Ytest,predicted)
        classe_1 = self.trainedX[self.trainedY == 1]
        classe_2 = self.trainedX[self.trainedY == -1]
        plt.figure('After Training')
        plt.scatter(classe_1[self.SelectedFeatures[0]], classe_1[self.SelectedFeatures[1]], c='green')
        plt.scatter(classe_2[self.SelectedFeatures[0]], classe_2[self.SelectedFeatures[1]], c='red')
        plt.xlabel(self.SelectedFeatures[0])
        plt.ylabel(self.SelectedFeatures[1])
        Xplot = [self.trainedX.min(), self.trainedX.max()]
        Yplot = 0
        # Y = (-w1 - bias) /w2
        if (self.bias == 1):
            Yplot = [(-self.weights[0][0] - (self.weights[0][1] * Xplot[0])) / self.weights[0][2],
                     (-self.weights[0][0] - (self.weights[0][1] * Xplot[1])) / self.weights[0][2]]
        else:
            Yplot = [(0 - (self.weights[0][0] * Xplot[0])) / self.weights[0][1],
                     (0 - (self.weights[0][0] * Xplot[1])) / self.weights[0][1]]
        plt.plot(Xplot, Yplot, color='k')
        plt.show()
        matrix = self.confusion_matrix(predicted, self.Y)
        print('TP --> '+str(matrix[0, 0]))
        print('FP --> '+str(matrix[0, 1]))
        print('FN --> '+str(matrix[1, 0]))
        print('TN --> '+str(matrix[1, 1]))
        return Accuracy

    def Error(self, predicted, actual):
        Error = predicted - actual
        return Error

    def Split_Data(self):
        self.Xtrain, self.Xtest, self.YTrain, self.Ytest = train_test_split(self.X, self.Y, test_size=0.4,
                                                                            train_size=0.6, shuffle=True,
                                                                            stratify=self.Y)

    def confusion_matrix(self, pred, original):
        matrix = np.zeros((2, 2))
        for i in range(len(pred)):
            # 1=positive, -1=negative
            if int(pred[i]) == 1 and int(original[i]) == 1:
                matrix[0, 0] += 1  # True Positives
            elif int(pred[i]) == -1 and int(original[i]) == 1:
                matrix[0, 1] += 1  # False Positives
            elif int(pred[i]) == 1 and int(original[i]) == -1:
                matrix[1, 0] += 1  # False Negatives
            elif int(pred[i]) == -1 and int(original[i]) == -1:
                matrix[1, 1] += 1  # True Negatives
        return matrix
