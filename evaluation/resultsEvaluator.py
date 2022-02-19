import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import sys


# example:
# c:\pycharmEnv\pythin37x64Env\Scripts\python C:\bgu\session-rec\evaluation\resultsEvaluator.py C:\bgu\session-rec\finalResults\rsc15_64_1EOS_LR\SGNN\ C:\bgu\session-rec\data\rsc15\prepared_\rsc15_64_1EOS\rsc15-clicks64_test.txt

def perRec(Y, results):
    precision, recall, thresholds = precision_recall_curve(Y, results)
    maxF1 = 0
    maxThreshold = 0
    maxI = 0

    maxF1_02 = 0
    maxThreshold_02 = 0
    maxI_02 = 0

    # todo: calc fp/fn/tp/tn hr
    # todo: merge a few graphs 1 - 10 - 100
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

    print(str(maxI) + ' F1 ' + str(maxF1) + ' t ' + str(maxThreshold) + ' p ' + str(precision[maxI]) + ' r ' + str(recall[maxI]))
    print(str(maxI_02) + '0.2 F1 ' + str(maxF1_02) + ' t ' + str(maxThreshold_02) + ' p ' + str(precision[maxI_02]) + ' r ' + str(recall[maxI_02]))
    return maxF1, maxThreshold, maxI


resDir = sys.argv[1]
# resDir = 'finalResults/diginetica_1EOS_LR/SGNN/'
testFile = sys.argv[2]
testFileReady = sys.argv[3]

# testFile  = 'data/diginetica/prepared/diginetica_1EOS/train-item-views_full_test.txt'
resultsFile = resDir + 'clfProbs.csv'
resultsFileBL = resDir + 'clfProbsBaseLine.csv'
results = pd.read_csv(resultsFile, sep='\t', dtype={'ItemId': np.int64}, header=None)
resultsBL = pd.read_csv(resultsFileBL, sep='\t', dtype={'ItemId': np.int64}, header=None)

if testFileReady == 'True':
    testData = pd.read_csv(testFile + '.Y.csv', sep='\t', dtype={'ItemId': np.int64}, header=None)
else:
    testData = pd.read_csv(testFile, sep='\t', dtype={'ItemId': np.int64})
    countSessionsInTest = testData.groupby(['SessionId']).count()
    sessions = testData.groupby(['SessionId']).count()
    if len(testData) - len(sessions) != len(results):
        print('Error - data Lens not fit;  test' + str(len(testData)) + ' sessions ' + str(len(sessions)) + ' = ' + str(len(testData) - len(sessions)) + ' != ' + str(len(results)))
        exit(0)
    else:
        print('data len OK')

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

    testData.to_csv(testFile + '.Y.csv', sep=";", header=False, index=False)

Y = testData.values.tolist()
print('results')
perRec(Y, results)
print('resultsBL')
perRec(Y, resultsBL)
print('done ')
