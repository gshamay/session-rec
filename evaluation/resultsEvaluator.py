import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

def perRec(Y,results):
    precision, recall, thresholds = precision_recall_curve(Y,results)
    maxF1 = 0
    maxThreshold = 0
    maxI = 0
    for i in range(len(thresholds)):
        r = recall[i]
        p = precision[i]
        F1 = (2 * r * p) / (r + p)
        if (F1 > maxF1):
            maxF1 = F1
            maxThreshold = thresholds[i]
            maxI = i
    print(str(maxI) + ' F1 ' + str(maxF1) + ' t ' + str(maxThreshold) + ' p ' + str(precision[maxI]) + ' r ' + str(recall[maxI]))
    return  maxF1, maxThreshold, maxI

resultsFile = 'finalResults/diginetica_1EOS_LR/SGNN/clfProbs.csv'
testFile = 'data/diginetica/prepared/diginetica_1EOS/train-item-views_full_test.txt'
results = pd.read_csv(resultsFile, sep='\t', dtype={'ItemId': np.int64}, header=None)
resultsFileBL = 'finalResults/diginetica_1EOS_LR/SGNN/clfProbsBaseLine.csv'
resultsBL = pd.read_csv(resultsFileBL, sep='\t', dtype={'ItemId': np.int64}, header=None)
testData = pd.read_csv(testFile, sep='\t', dtype={'ItemId': np.int64})
countSessionsInTest = testData.groupby(['SessionId']).count()
sessions = testData.groupby(['SessionId']).count()
if len(testData) - len(sessions) != len(results):
    print('Error - data Lens not fit')
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
Y = testData.values.tolist()

print('results')
perRec(Y,results)
print('resultsBL')
perRec(Y,resultsBL)
print('done ' + str(sessionCount))
