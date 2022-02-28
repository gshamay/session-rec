import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import sys
import matplotlib.pyplot as plt


# parameters:
# 1. resuls directory - resultsFile contain 'clfProbs.csv' and 'clfProbsBaseLine.csv'
# for example C:\bgu\session-rec\finalResults\rsc15_64_1EOS_LR\SGNN\
# 2. Test Data File Full Path
# for example: C:\bgu\session-rec\data\rsc15\prepared_\rsc15_64_1EOS\rsc15-clicks64_test.txt
# 3. True - use Y test data file
# if this evaluation process run on the data - it generate a Y.csv file near the data file - for example rsc15-clicks64_test.txt.Y.csv
# This file contain TRE/FALSE values, derived from the test data,
# which is the actual classification of the test data to be compared with the results provided in parameter 1

# example:
# c:\pycharmEnv\pythin37x64Env\Scripts\python.exe C:\bgu\session-rec\evaluation\resultsEvaluator.py C:\bgu\session-rec\finalResults\rsc15_64_1EOS_LR\SGNN\ C:\bgu\session-rec\data\rsc15\prepared_\rsc15_64_1EOS\rsc15-clicks64_test.txt True


def perRec(Y, results):
    # todo: merge a few graphs 1 - 10 - 100
    res = results.values.flatten().tolist()
    precision, recall, thresholds = precision_recall_curve(Y, res)
    # thresholds = [0.2,0.25,0.3,0.35]
    # classificationPerTresholdEvaluation(Y, results, precision, recall, thresholds)
    return precision, recall, thresholds


def classificationPerTresholdEvaluation(Y, results, precision, recall, thresholds):
    # todo: calc fp/fn/tp/tn/f1
    maxF1 = 0
    maxThreshold = 0
    maxI = 0
    maxF1_02 = 0
    maxThreshold_02 = 0
    maxI_02 = 0

    minTH = 1
    maxTH = 0
    for i in range(len(thresholds)):
        r = recall[i]
        p = precision[i]
        t = thresholds[i]
        F1 = (2 * r * p) / (r + p)
        if (F1 > maxF1):
            maxF1 = F1
            maxThreshold = t
            maxI = i

        if (t < 0.8 and t > 0.2):
            if (F1 > maxF1_02):
                maxF1_02 = F1
                maxThreshold_02 = t
                maxI_02 = i

        # f1_score_ = f1_score(Y, resBoolArr)
        # if(F1 != f1_score_):
        #     print ('wrong!')

        if minTH > t:
            minTH = t
        if maxTH < t:
            maxTH = t

        doCalcTPTNFPFN = True
        if (doCalcTPTNFPFN):
            calcTPTNFPFN(Y, results, t)

    print('minTH ' + str(minTH)
          + ' maxTH ' + str(maxTH))
    print('maxI ' + str(maxI) + ' F1 ' + str(maxF1) + ' th ' + str(maxThreshold) + ' p ' + str(precision[maxI]) + ' r ' + str(recall[maxI]))
    print('maxI02 ' + str(maxI_02) + ' 0.2 F1 ' + str(maxF1_02) + ' th ' + str(maxThreshold_02) + ' p ' + str(precision[maxI_02]) + ' r ' + str(recall[maxI_02]))


def calcTPTNFPFN(Y, results, t):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    resBoolArr = results.apply(lambda x: x >= t).values.flatten()
    for vY, vRes in zip(Y, resBoolArr):
        resBool = vY == vRes
        if resBool:
            if (vY):
                TP += 1
            else:
                TN += 1
        else:
            if (vRes):
                FP += 1
            else:
                FN += 1

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    print('TH' + str(t)
          + '\t' + ' TP ' + str(TP)
          + '\t' + ' TN ' + str(TN)
          + '\t' + ' FP ' + str(FP)
          + '\t' + ' FN ' + str(FN)
          + '\t' + ' len ' + str(len(Y))
          + '\t' + ' recall ' + str(recall)
          + '\t' + ' precision ' + str(precision)
          )


resultsFilesDir = []
names = []
dataLocation = None
testFileReady = True
fileName = ''

def printGraphs():
    global index
    print(resultsFilesDir)
    print(dataLocation)
    if testFileReady:
        testData = pd.read_csv(dataLocation + '.Y.csv', sep='\t', header=None)
    else:
        testData = pd.read_csv(dataLocation, sep='\t', dtype={'ItemId': np.int64})
        session_key = 'SessionId'
        time_key = 'Time'
        testData.sort_values([session_key, time_key], inplace=True)
        countSessionsInTest = testData.groupby(['SessionId']).count()
        sessions = testData.groupby(['SessionId']).count()

        # if len(testData) - len(sessions) != len(results):
        #     print('Error - data Lens not fit;  test' + str(len(testData)) + ' sessions ' + str(len(sessions)) + ' = ' + str(len(testData) - len(sessions)) + ' != ' + str(len(results)))
        #     exit(0)
        # else:
        #     print('data len OK')

        testData = testData[['SessionId', 'ItemId']]
        testData1 = testData.copy()
        sessionCount = 0
        currentSessionId = 0
        for index, row in testData.iterrows():
            sessionId = row['SessionId']
            if (sessionId != currentSessionId):
                # print(str(sessionId ))
                currentSessionId = sessionId
                sessionCount += 1
                if (sessionCount % 1000 == 0):
                    print(str(sessionCount))

                testData.loc[index, ['SessionId']] = 0

        testData.drop(testData.index[testData['SessionId'] == 0], inplace=True)
        testData2 = testData.copy()
        testData = testData['ItemId'].apply(lambda x: x == -1)
        testData.to_csv(dataLocation + '.Y.csv', sep=";", header=False, index=False)
    plt.figure(0, clear=True)
    Y = testData.values.flatten().tolist()
    # Y = list(reversed(Y))
    plotStyle = [
        '-',  # solid line style
        '--',  # dashed line style
        '-.',  # dash-dot line style
        ':'  # dotted line style
    ]
    plotStyleIdx = 0
    bBaseLineEvaluadted = False
    if not bBaseLineEvaluadted:
        resDir = resultsFilesDir[0]
        resultsFileBL = resDir + 'clfProbsBaseLine.csv'  # baseline is similare to all aEOS sizes
        resultsBL = pd.read_csv(resultsFileBL, sep='\t', dtype={'ItemId': np.int64}, header=None)
        print('resultsBL')
        bBaseLineEvaluadted = True
        precision, recall, thresholds = perRec(Y, resultsBL)

        while (len(thresholds) < len(precision)):
            precision = np.delete(precision, len(precision) - 1)

        while (len(thresholds) < len(recall)):
            recall = np.delete(recall, len(recall) - 1)

        while (len(thresholds) > len(precision)):
            thresholds = np.delete(thresholds, len(thresholds) - 1)

        plt.figure(0, clear=True)
        plt.plot(thresholds, precision, label="precision")
        plt.plot(thresholds, recall, label="recall")
        plt.xlabel('Threshold')
        plt.ylabel('Precision,Recall')
        plt.legend()
        plt.savefig(dataLocation + ' BaseLineThresholds.png')

        limit = 0.00
        precision = precision[recall > limit]
        thresholds = thresholds[recall > limit]
        recall = recall[recall > limit]
        # limit = 1.00
        # precision = precision[recall < limit]
        # recall = recall[recall < limit]
        # recall = recall[precision < limit]
        # precision = precision[precision < limit]

        plt.figure(1, clear=True)
        plt.plot(recall, precision, plotStyle[plotStyleIdx], label="Baseline")
        plotStyleIdx += 1
        plotStyleIdx = plotStyleIdx % len(plotStyle)
    namesIdx = 0
    for resDir in resultsFilesDir:
        resultsFile = resDir + 'clfProbs.csv'  # the current classifier
        results = pd.read_csv(resultsFile, sep='\t', header=None)
        precision, recall, thresholds = perRec(Y, results)

        # limit = 0.00
        # precision = precision[recall > limit]
        # recall = recall[recall > limit]
        # recall = recall[precision > limit]
        # precision = precision[precision > limit]
        #
        # limit = 1.00
        # precision = precision[recall < limit]
        # recall = recall[recall < limit]
        # recall = recall[precision < limit]
        # precision = precision[precision < limit]

        plt.plot(recall, precision, plotStyle[plotStyleIdx], label=names[namesIdx])
        namesIdx += 1
        plotStyleIdx += 1
        plotStyleIdx = plotStyleIdx % len(plotStyle)
    # ZOOM
    # plt.axis([0, None, 0, max(precision)])  # plt.axis([x_min, x_max, y_min, y_max])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(fileName + '.png')


def debug_shortFiles():
    pass
    # Debug - Short Files
    # resultsFilesDir.append("C:/bgu/session-rec/results/diginetica/diginetica_Short_1EOS_LR/AR/")
    # dataLocation = 'C:/bgu/session-rec/data/diginetica/prepared/diginetica_1EOS_short/train-item-views_full_test.txt'


def RR0All():
    global resultsFilesDir
    global names
    global fileName
    global dataLocation
    resultsFilesDir = []
    names = []
    dataLocation = 'C:/bgu/session-rec/data/retailrocket/slices/1EOS/events_test.0.txt'
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/retailrocket_1EOS_W0_LR/CSRM/")
    names.append('CSRM')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/retailrocket_1EOS_W0_LR/SGNN/")
    names.append('SR-GNN')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/retailrocket_1EOS_W0_LR/VSKNN/")
    names.append('VSKNN')
    fileName = 'RR0_1EOS_all'
    printGraphs()

def RR3All():
    global resultsFilesDir
    global names
    global fileName
    global dataLocation
    resultsFilesDir = []
    names = []
    dataLocation = 'C:/bgu/session-rec/data/retailrocket/slices/1EOS/events_test.2.txt'
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/retailrocket_1EOS_W3_LR/CSRM/")
    names.append('CSRM')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/retailrocket_1EOS_W3_LR/SGNN/")
    names.append('SR-GNN')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/retailrocket_1EOS_W3_LR/VSKNN/")
    names.append('VSKNN')
    fileName = 'RR3_1EOS_all'
    printGraphs()

def RR5All():
    global resultsFilesDir
    global names
    global fileName
    global dataLocation
    resultsFilesDir = []
    names = []
    dataLocation = 'C:/bgu/session-rec/data/retailrocket/slices/1EOS/events_test.4.txt'
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/retailrocket_1EOS_W5_LR/CSRM/")
    names.append('CSRM')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/retailrocket_1EOS_W5_LR/SGNN/")
    names.append('SR-GNN')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/retailrocket_1EOS_W5_LR/VSKNN/")
    names.append('VSKNN')
    fileName = 'RR5_1EOS_all'
    printGraphs()


def digiAll():
    global resultsFilesDir
    global names
    global fileName
    global dataLocation
    resultsFilesDir = []
    names = []
    dataLocation  = 'C:/bgu/session-rec/data/diginetica/prepared/diginetica_1EOS/train-item-views_full_test.txt'

    resultsFilesDir.append("C:/bgu/session-rec/finalResults/diginetica_10EOS_LR/CSRM/")
    names.append('CSRM')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/diginetica_1EOS_LR/SGNN/")
    names.append('SR-GNN')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/diginetica_1EOS_LR/VSKNN/")
    names.append('VSKNN')
    fileName = 'digi_1EOS_all'
    printGraphs()


def digiSGNN():
    global resultsFilesDir
    global names
    global fileName
    global dataLocation
    resultsFilesDir = []
    names = []
    dataLocation  = 'C:/bgu/session-rec/data/diginetica/prepared/diginetica_1EOS/train-item-views_full_test.txt'

    resultsFilesDir.append("C:/bgu/session-rec/finalResults/diginetica_1EOS_LR/SGNN/")
    names.append('1 aEOS')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/diginetica_10EOS_LR/SGNN/")
    names.append('10 aEOS')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/diginetica_100EOS_LR/SGNN/")
    names.append('100 aEOS')
    fileName = 'digi_1-10-100-gnn'
    printGraphs()

def digiCSRM():
    global resultsFilesDir
    global names
    global fileName
    global dataLocation
    resultsFilesDir = []
    names = []
    dataLocation = 'C:/bgu/session-rec/data/diginetica/prepared/diginetica_1EOS/train-item-views_full_test.txt'
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/diginetica_10EOS_LR/CSRM/")
    names.append('10 aEOS')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/diginetica_100EOS_LR/CSRM/")
    names.append('100 aEOS')
    fileName = 'digi_10-100-CSRM'
    printGraphs()


def rscAll():
    global resultsFilesDir
    global names
    global fileName
    global dataLocation
    resultsFilesDir = []
    names = []
    dataLocation = 'C:/bgu/session-rec/data/rsc15/prepared/rsc15_64_1EOS/rsc15-clicks64_test.txt'

    resultsFilesDir.append("C:/bgu/session-rec/finalResults/rsc15_64_1EOS_LR/CSRM/")
    names.append('CSRM')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/rsc15_64_1EOS_LR/SGNN/")
    names.append('SR-GNN')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/rsc15_64_1EOS_LR/VSKNN/")
    names.append('VSKNN')
    fileName = 'rsc_1EOS_all'
    printGraphs()


def rscSGNN():
    global resultsFilesDir
    global names
    global fileName
    global dataLocation
    resultsFilesDir = []
    names = []
    dataLocation = 'C:/bgu/session-rec/data/rsc15/prepared/rsc15_64_1EOS/rsc15-clicks64_test.txt'

    resultsFilesDir.append("C:/bgu/session-rec/finalResults/rsc15_64_1EOS_LR/SGNN/")
    names.append('1 aEOS')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/rsc15_64_10EOS_LR/SGNN/")
    names.append('10 aEOS')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/rsc15_64_100EOS_LR/SGNN/")
    names.append('100 aEOS')
    fileName = 'rsc_1-10-100-gnn'
    printGraphs()

def rscCSRM():
    global resultsFilesDir
    global names
    global fileName
    global dataLocation
    resultsFilesDir = []
    names = []
    dataLocation = 'C:/bgu/session-rec/data/rsc15/prepared/rsc15_64_1EOS/rsc15-clicks64_test.txt'

    resultsFilesDir.append("C:/bgu/session-rec/finalResults/rsc15_64_1EOS_LR/CSRM/")
    names.append('1 aEOS')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/rsc15_64_10EOS_LR/CSRM/")
    names.append('10 aEOS')
    resultsFilesDir.append("C:/bgu/session-rec/finalResults/rsc15_64_100EOS_LR/CSRM/")
    names.append('100 aEOS')
    fileName = 'rsc_1-10-100-CSRM'
    printGraphs()

if (len(sys.argv) <= 1):
    ######################################################
    RR0All()
    RR3All()
    RR5All()

    rscAll()
    rscSGNN()
    rscCSRM()

    digiAll()
    digiSGNN()
    digiCSRM()
    #debug_shortFiles()
else:
    # Read values from the cmd
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
    # Use like: # python arg.py -l 1234 2345 3456 4567
    # parser.parse_args()
    resultsFilesDir.append(sys.argv[1])
    # resDir = 'finalResults/diginetica_1EOS_LR/SGNN/'
    dataLocation = sys.argv[2]
    testFileReady = False
    printGraphs()

print('done ')