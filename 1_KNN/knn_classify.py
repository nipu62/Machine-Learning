'''
@author: Ayesha Siddika Nipu (an37s)
Description: This program implements k-nearest neighbor classification
Command to run: python knn_classify.py pendigits_training.txt pendigits_test.txt <k>
'''

from sys import argv
from math import sqrt
from random import choice

#   Returns list of mean and standard deviation for each column
def calculateMeanAndStandardDeviation(data):
    meanList = []
    stdList = []
    for i in range(len(data[0]) - 1):
        col = [row[i] for row in data]
        mean = sum(col) / len(col)
        sigma = 0
        for each in col:
            val = abs((each - mean))
            sigma += val * val
        std = sqrt(sigma / len(col))
        meanList.append(mean)
        stdList.append(std)
    #print("MeanList: ", meanList, "\nStdList: ", stdList)

    return meanList, stdList

#   Normalizes each data using F(v) = (v−mean) / std
def normalizeDataset(meanList, stdList, data):
    length = len(data)
    #print('length: ', length, '\n')
    for i in range(length):
        for j in range(len(data[i]) -1):
            mean = meanList[j]
            std = stdList[j]
            data[i][j] = (data[i][j] - mean)/std
    #print("Normalized data: ", data, "\n\n")

#   Returns euclidian distance between two row
def calculateEuclidianDistance(row1, row2):
    dist = 0.0
    length = len(row1)
    for i in range(length - 1):
        dist += (row1[i] - row2[i]) * (row1[i] - row2[i])
    return sqrt(dist)

#   Returns nearest neibours comparing the shortest Euclidian Distance
def calculateNeighbour(trainingData, testRow, k):
    distanceList = list()
    for trainingRow in trainingData:
        dist = calculateEuclidianDistance(trainingRow, testRow)
        distanceList.append([trainingRow, dist])

    #print(distanceList)

    distanceList.sort(key=lambda row: row[1])
    neighbour = list()
    for i in range(int(k)):
        neighbour.append(distanceList[i][0])
    return neighbour

#   Returns the class predicted by KNN
def predictClass(trainingData, testRow, k):
    neighbors = calculateNeighbour(trainingData, testRow, k)
    output_vals = [row[-1] for row in neighbors]
    counts = dict()
    for i in output_vals:
        counts[i] = counts.get(i, 0) + 1
    v = [value for value in counts.values()]

    # Choosing a random class if there is a tie
    prediction = choice([key for key in counts if counts[key] == max(v)])

    return prediction

#   Returns a count dictionary for accuracy measurement
def calculateCount(trainingData, testRow, k):
    neighbors = calculateNeighbour(trainingData, testRow, k)
    output_vals = [row[-1] for row in neighbors]
    counts = dict()
    for i in output_vals:
        counts[i] = counts.get(i, 0) + 1

    return counts

#   Prints object ID, prediction class and true class as per instruction
def calculateAccuracy(trainData, testData, k):
    classification_accuracy = 0
    length_testData = len(testData)
    for obj_id in range(length_testData):
        row = testData[obj_id]
        pred_class = predictClass(trainData, row, k)
        true_class = row[-1]
        accuracy = 0

        counts = calculateCount(trainData, row, k)
        v = [value for value in counts.values()]

        if (v.count(max(v)) == 1 and (pred_class == true_class)):
            accuracy = 1
        elif (v.count(max(v)) > 1 and counts[pred_class] == max(v)):
            accuracy = 1 / v.count(max(v))

        #print("ID={0:5d}, predicted={1:3d}, true={2:3d}\n".format(obj_id, pred_class, true_class))
        classification_accuracy += accuracy

    classification_accuracy = classification_accuracy / len(testData)
    print("classification accuracy= {0:6.4f}\n".format(classification_accuracy))


def main():
    script, pendigits_training, pendigits_test, k = argv

    file_train = open(pendigits_training, 'r')
    file_test = open(pendigits_test, 'r')

    training_dataset = []
    for row in file_train:
        temp = []
        for each in row.split():
            temp.append(int(each))
        training_dataset.append(temp)
    print("training_dataset: ", len(training_dataset), '\n')

    test_dataset = []
    for row in file_test:
        temp = []
        for each in row.split():
            temp.append(int(each))
        test_dataset.append(temp)
    print("test_dataset: ", len(test_dataset), '\n')

    mean, std = calculateMeanAndStandardDeviation(training_dataset)

    normalizeDataset(mean, std, training_dataset)
    #print("Training dataset: ", training_dataset, "\n\n")

    normalizeDataset(mean, std, test_dataset)
    #print("Test dataset: ", test_dataset, "\n\n")

    calculateAccuracy(training_dataset, test_dataset, k)

    file_train.close()
    file_test.close()

if __name__ == '__main__':
    main()

