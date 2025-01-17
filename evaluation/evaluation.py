import time
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import PrecisionRecallDisplay, f1_score
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def evaluate_sessions_batch(pr, metrics, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time', batch_size=100, break_ties=True):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    pr : gru4rec.GRU4Rec
        A trained instance of the GRU4Rec network.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    break_ties : boolean
        Whether to add a small random number to each prediction value in order to break up possible ties, which can mess up the evaluation. 
        Defaults to False, because (1) GRU4Rec usually does not produce ties, except when the output saturates; (2) it slows down the evaluation.
        Set to True is you expect lots of ties.
    
    Returns
    --------
    out : list of tuples
        (metric_name, value)
    
    '''

    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    print('START batch eval ', actions, ' actions in ', sessions, ' sessions')
    sc = time.perf_counter()
    st = time.time()

    for m in metrics:
        m.reset()

    pr.predict = None #In case someone would try to run with both items=None and not None on the same model without realizing that the predict function needs to be replaced
    test_data.sort_values([session_key, time_key], inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()

    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1

    iters = np.arange(batch_size).astype(np.int32)

    maxiter = iters.max()
    start = offset_sessions[iters]
    end = offset_sessions[iters+1]

    in_idx = np.zeros(batch_size, dtype=np.int32)
    np.random.seed(42)

    while True:

        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break

        start_valid = start[valid_mask]
        minlen = (end[valid_mask]-start_valid).min()
        in_idx[valid_mask] = test_data[item_key].values[start_valid]

        for i in range(minlen-1):

            out_idx = test_data[item_key].values[start_valid+i+1]

            if items is not None:
                uniq_out = np.unique(np.array(out_idx, dtype=np.int32))
                preds = pr.predict_next_batch(iters, in_idx, np.hstack([items, uniq_out[~np.in1d(uniq_out,items)]]), batch_size)
            else:
                preds = pr.predict_next_batch(iters, in_idx, None, batch_size)

            preds.fillna(0, inplace=True)
            in_idx[valid_mask] = out_idx

#             for m in metrics:
#                 m.add_batch( preds.loc[:,valid_mask], out_idx )

            i=0
            for part, series in preds.loc[:,valid_mask].iteritems():
                preds.sort_values( part, ascending=False, inplace=True )
                for m in metrics:
                    m.add( preds[part], out_idx[i] )
                i += 1

        start = start+minlen-1
        mask = np.arange(len(iters))[(valid_mask) & (end-start<=1)]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions)-1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter+1]

    print( 'END batch eval ', (time.perf_counter()-sc), 'c / ', (time.time()-st), 's' )

    res = []
    for m in metrics:
        res.append( m.result() )

    return res

def evaluate_sessions_batch_org(pr, metrics, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time', batch_size=100, break_ties=True ):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    pr : gru4rec.GRU4Rec
        A trained instance of the GRU4Rec network.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    break_ties : boolean
        Whether to add a small random number to each prediction value in order to break up possible ties, which can mess up the evaluation. 
        Defaults to False, because (1) GRU4Rec usually does not produce ties, except when the output saturates; (2) it slows down the evaluation.
        Set to True is you expect lots of ties.
    
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)
    
    '''

    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    print('START batch eval old ', actions, ' actions in ', sessions, ' sessions')
    sc = time.perf_counter()
    st = time.time()

    pr.predict = None #In case someone would try to run with both items=None and not None on the same model without realizing that the predict function needs to be replaced
    test_data.sort_values([session_key, time_key], inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    evalutation_point_count = 0
    mrr, recall = 0.0, 0.0
    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1

    iters = np.arange(batch_size).astype(np.int32)
    maxiter = iters.max()

    start = offset_sessions[iters]
    end = offset_sessions[iters+1]

    in_idx = np.zeros(batch_size, dtype=np.int32)

    np.random.seed(42)
    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]

        minlen = (end[valid_mask]-start_valid).min()
        in_idx[valid_mask] = test_data[item_key].values[start_valid]
        for i in range(minlen-1):

            out_idx = test_data[item_key].values[start_valid+i+1]

            if items is not None:
                uniq_out = np.unique(np.array(out_idx, dtype=np.int32))
                preds = pr.predict_next_batch(iters, in_idx, np.hstack([items, uniq_out[~np.in1d(uniq_out,items)]]), batch_size)
            else:
                preds = pr.predict_next_batch(iters, in_idx, None, batch_size)

            if break_ties:
                preds += np.random.rand(*preds.values.shape) * 1e-8

            preds.fillna(0, inplace=True)
            in_idx[valid_mask] = out_idx

            if items is not None:
                others = preds.ix[items].values.T[valid_mask].T
                targets = np.diag(preds.ix[in_idx].values)[valid_mask]
                ranks = (others > targets).sum(axis=0) +1
            else:
                ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
                targets = np.diag(preds.ix[in_idx].values)[valid_mask]

            rank_ok = ranks < cut_off
            recall += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evalutation_point_count += len(ranks)

        start = start+minlen-1
        mask = np.arange(len(iters))[(valid_mask) & (end-start<=1)]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions)-1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter+1]

    print( 'END batch eval old', (time.perf_counter()-sc), 'c / ', (time.time()-st), 's' )
    print( 'hit rate ', recall/evalutation_point_count )
    print( 'mrr ', mrr/evalutation_point_count )

    return recall/evalutation_point_count, mrr/evalutation_point_count


def evaluate_sessions(pr, metrics, test_data_, train_data, algorithmKey, conf, items=None, cut_off=20,
                      session_key='SessionId', item_key='ItemId', time_key='Time'):
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out :  list of tuples
        (metric_name, value)
    
    '''

    for m in metrics:
        m.reset()

    try:
        train_data = train_data.drop(columns='index')
    except KeyError:
        print('Error - KeyError - train_data = train_data.drop( columns=index) failed!')
    except Exception:
        print('Error - Exception - train_data = train_data.drop( columns=index) failed!')

    # do evaluation on both train and test
    #  This is required for generating data that is readable later from the csv file for the usage of LR

    bRunningTrainData = False
    dataToTest = [test_data_]
    if 'useBothTrainAndTest' in conf['results'] and conf['results']['useBothTrainAndTest'] is True:
        # when useBothTrainAndTest avoid calculating statistics for train_data - test data must be SECOND !!!
        dataToTest = [train_data, test_data_]
        bRunningTrainData = True
        print('running vealuation on Both Train And Test - only for filling csv results')
        print('train Size[' + str(len(train_data)) + ']')

    print('test Size[' + str(len(test_data_)) + ']')
    LRTestX = None
    LRTestY = None
    failedPredictions  = 0
    for test_data in dataToTest:  # create predictions on either test only or on train AND test
        if(bRunningTrainData):
            print("start running evaluation on train")
        else:
            print("start running evaluation on test")

        actions = len(test_data)
        sessions = len(test_data[session_key].unique())
        count = 0
        print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

        sc = time.perf_counter()
        st = time.time()

        time_sum = 0
        time_sum_clock = 0
        time_count = 0

        test_data.sort_values([session_key, time_key], inplace=True)
        test_data.to_csv(conf['results']['folder'] + 'TestDataSorted.csv', sep=";", header=False, index=False)

        items_to_predict = train_data[item_key].unique()
        prev_iid, prev_sid = -1, -1
        pos = 0

        # LR #############################
        runLR = False
        clf = None
        clfBaseLine = None
        if 'clf' in conf and 'clfBaseLine' in conf:
            # use the LR models (main and baseline) in case they were created in the run_config
            clfBaseLine = conf['clfBaseLine']
            clf = conf['clf']

        if clf is not None \
                and clfBaseLine is not None \
                and not bRunningTrainData:
            # avoid getting LR info on train data
            # enable LR only when it's available
            runLR = True

        LRTestX = []
        LRTestY = []
        #######################################
        for i in range(len(test_data)):
            if count % 1000 == 0:
                print( '    eval process: ', count, ' of ', actions, ' actions: ', ( count / actions * 100.0 ), ' % in',(time.time()-st), 's')

            sid = test_data[session_key].values[i]
            iid = test_data[item_key].values[i]
            ts = test_data[time_key].values[i]
            if prev_sid != sid:
                prev_sid = sid
                pos = 0
            else:
                if items is not None:
                    if np.in1d(iid, items): items_to_predict = items
                    else: items_to_predict = np.hstack(([iid], items))

                crs = time.perf_counter()
                trs = time.time()

                for m in metrics:
                    if hasattr(m, 'start_predict'):
                        # only some of the metrics, as the running time metric (Time_usage_training), has 'start_predict' variable
                        m.start_predict( pr )

                doContinue = False
                try:
                    # predict all sub sessions
                    preds = pr.predict_next(sid, prev_iid, items_to_predict, timestamp=ts)
                except Exception:
                    doContinue = True
                except IndexError:
                    doContinue = True

                if doContinue:
                    failedPredictions += 1
                    pos += 1
                    prev_iid = iid
                    count += 1
                    if (failedPredictions % 10 == 0):
                        print('failedPredictions.predict_next[' + str(failedPredictions) + ']')
                    continue

                # preds contain now a list of all possible items with their probabilities to be the next item

                for m in metrics:
                    if hasattr(m, 'stop_predict'):
                        # same as 'start_predict' above - relevant only for some of the metrics
                        m.stop_predict(pr)

                # todo: refactor. Duplicated code that is handled differently between the
                #  different evaluate_sessions methods;  Also appear as preds.fillna(0, inplace=True)

                preds[np.isnan(preds)] = 0
                # in case that some prediction was not a valid number (NaN) -it's probability is zeroed

                #  preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
                preds.sort_values( ascending=False, inplace=True ) # sort preds according to the predicted probability

                ############################################################
                # Handle multiple aEOS predictions
                #  in case there are more then a single aEOS, we need to select the top one,
                #  set it's value to -1, and push all the rest down as they are not relevant
                #  change all -1...-N to be -1
                #  'pushUp' all other results to cover the unneeded -2...-N
                #  this will keep only  the -1 with the highest probability value
                aEOSBaseIDValue = -1
                aEOSMaxPredictedValue = 0
                defaultMinValueToPushDownPrediction = -0.01

                # Method 1
                # eval process:  1000  of  3006  actions:  33.266799733865604  % in 10.690671920776367 s
                foundAEOS = False
                maxUsedK = 50
                for i in range(maxUsedK):
                    iKey = preds.index[i]
                    if (iKey <= aEOSBaseIDValue):
                        if (not foundAEOS):
                            foundAEOS = True
                            aEOSMaxPredictedValue = preds[iKey]
                        preds[iKey] = defaultMinValueToPushDownPrediction  # push the result down the results list

                if (aEOSMaxPredictedValue > 0):
                    preds[aEOSBaseIDValue] = aEOSMaxPredictedValue
                    # re sort preds according to the new  predicted probabilities
                    preds.sort_values(ascending=False, inplace=True)

                # if the test actual data next id is aEOS with id < -1 (-2, -3....-N),
                # we change it to -1 as it's the default aEOS id
                if (iid <= aEOSBaseIDValue):
                    iid = aEOSBaseIDValue

                ############################################################
                # Prepare data for LR [[aEOSMaxPredictedValue,sessionLen]...]
                if runLR:
                    LRTestX.append([aEOSMaxPredictedValue, pos])
                    LRTestY.append(iid <= aEOSBaseIDValue)
                ############################################################

                time_sum_clock += time.perf_counter() - crs
                time_sum += time.time() - trs
                time_count += 1

                for m in metrics:
                    if bRunningTrainData and type(m).__name__ != 'Saver':
                        # only save metrics should be active when evaluating train data
                        # for saving it in the csv
                        # other metrics must be avoided
                        continue

                    if hasattr(m, 'add'):
                        m.add(preds, iid, for_item=prev_iid, session=sid, position=pos)

                pos += 1

            prev_iid = iid
            count += 1

        print('evaluation Model ; errors In predict_next[' + str(failedPredictions) + ']')
        print('END evaluation in ', (time.perf_counter() - sc), 'c / ', (time.time() - st), 's')
        print('    avg rt ', (time_sum / time_count), 's / ', (time_sum_clock / time_count), 'c')
        print('    time count ', (time_count), 'count/', (time_sum), ' sum')

        if(bRunningTrainData):
            print("finished running evaluation on train")
        else:
            print("finished running evaluation on test")


        bRunningTrainData = False
        #  end for test_data in dataToTest



    res = []
    #############################################
    # Predict with LR and produce evaluation values: AUC, HR, FP, FN
    if runLR:
        clfProbs = clf.predict_proba(LRTestX)
        clfProbs = list(map(lambda x: x[1], clfProbs))
        clfProbdf = pd.DataFrame(clfProbs)
        clfProbdf.to_csv(conf['results']['folder'] + 'clfProbs.csv', sep=";", header=False, index=False)
        printLREvaluationValues("", LRTestY, clfProbs, res)

        LRTestYdf = pd.DataFrame(LRTestY)
        LRTestYdf.to_csv(conf['results']['folder'] + 'LRTestYdf.csv', sep=";", header=False, index=False)

        precision, recall, thresholds = precision_recall_curve(LRTestY, clfProbs)
        plt.figure(0, clear=True)
        plt.plot(recall, precision, label="Model")

        ################################################

        LRTestXBaseLine = list(map(lambda x: [x[1]], LRTestX))
        clfProbsBaseLine = clfBaseLine.predict_proba(LRTestXBaseLine)
        clfProbsBaseLine = list(map(lambda x: x[1], clfProbsBaseLine))
        clfProbsBaseLinedf = pd.DataFrame(clfProbsBaseLine)
        clfProbsBaseLinedf.to_csv(conf['results']['folder'] + 'clfProbsBaseLine.csv', sep=";",header=False,index=False)

        printLREvaluationValues("BaseLine", LRTestY, clfProbsBaseLine, res)

        precisionBaseLine, recallBaseLine, thresholdsBaseLine = precision_recall_curve(LRTestY, clfProbsBaseLine)
        plt.plot(recallBaseLine, precisionBaseLine, label="Baseline")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        name = conf['algorithms'][0]['key']+'_' + conf['data']['name']
        plt.legend()
        plt.savefig(conf['results']['folder'] + 'plot_PrecisionRecall_' + name + '.png')


        # todo: Check why we get the Pr/Rec values > thresholds
        while (len(thresholds) < len(precision)):
            precision = np.delete(precision, len(precision) - 1)

        while (len(thresholds) < len(recall)):
            recall = np.delete(recall, len(recall) - 1)

        while (len(thresholds) > len(precision)):
            thresholds = np.delete(thresholds, len(thresholds) - 1)

        thresholdsPrecisionRecall = np.array([thresholds, precision, recall])
        np.savetxt(conf['results']['folder'] + 'ThresholdsPrecisionRecall_' + name + '.csv', thresholdsPrecisionRecall, delimiter=";")

        plt.figure(1, clear=True)
        plt.plot(thresholds, precision, label="precision")
        plt.plot(thresholds, recall, label="recall")
        plt.xlabel('Threshold')
        plt.ylabel('Precision,Recall')
        plt.legend()
        plt.savefig(conf['results']['folder'] + 'plot_PrecisionRecallThresholds_' + name + '.png')
        # plt.show()  # Avoid showing plt - it hang the process

        stat, p = wilcoxon(x=clfProbs, y=clfProbsBaseLine)
        res.append(("Pvalue", str(p)))
        res.append(("stat", str(stat)))

    ############################################# if LR End
    plt.close()
    plt.cla()
    plt.clf()

    for m in metrics:
        if type(m).__name__ == 'Time_usage_testing':
            res.append(m.result_second(time_sum_clock / time_count))
            res.append(m.result_cpu(time_sum_clock / time_count))
        else:
            res.append(m.result())

    return res


def printLREvaluationValues(prefix, LRTestY, clfProbs, res):
    AUC = roc_auc_score(LRTestY, clfProbs)
    print('AUC' + prefix + '[' + str(AUC) + ']')
    res.append(("AUC" + prefix, str(AUC)))
    totalProbs = len(clfProbs)
    maxHR = 0
    maxHRAcc = 0
    posToPushMax = len(res)
    for accuracyMul in range(9):
        acc = accuracyMul * 0.1 + 0.1
        checkAcc = lambda t: t >= acc  # t >= acc --> True else False
        vfuncCheckAcc = np.vectorize(checkAcc)
        clfInAcc = vfuncCheckAcc(clfProbs)

        HR = 0
        FP = 0
        FN = 0
        for predictedWithAcc, y in zip(clfInAcc, LRTestY):
            if (predictedWithAcc == y):
                HR += 1
            else:
                if y:
                    FN += 1
                else:
                    FP += 1

        accStr = ("%.1f" % acc)
        HR = (HR / totalProbs)
        FP = (FP / totalProbs)
        FN = (FN / totalProbs)
        print('ACC' + prefix + '=' + accStr + ':'
              + 'HR' + prefix + '=' + str(HR) + ';'
              + 'FP' + prefix + '=' + str(FP) + ';'
              + 'FN' + prefix + '=' + str(FN)
              )
        if (maxHR < HR):
            maxHR = HR
            maxHRAcc = acc

        # todo: Create a method to add results
        res.append(("Hr" + prefix + "@" + accStr, str(HR)))
        res.append(("Fp" + prefix + "@" + accStr, str(FP)))
        res.append(("Fn" + prefix + "@" + accStr, str(FN)))
    res.insert(posToPushMax, ("maxHR" + prefix + "@" + str(("%.1f" % maxHRAcc)), str(maxHR)))
    return AUC, totalProbs


def evaluate_sessions_org(pr, metrics, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time'):
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)
    
    '''

    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    count = 0
    print('START org evaluation of ', actions, ' actions in ', sessions, ' sessions')
    st, sc = time.time(), time.perf_counter()

    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    evalutation_point_count = 0
    prev_iid, prev_sid = -1, -1
    mrr, recall = 0.0, 0.0
    for i in range(len(test_data)):

        if count % 1000 == 0:
            print( '    eval process: ', count, ' of ', actions, ' actions: ', ( count / actions * 100.0 ), ' % in',(time.time()-st), 's')

        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        if prev_sid != sid:
            prev_sid = sid
        else:
            if items is not None:
                if np.in1d(iid, items): items_to_predict = items
                else: items_to_predict = np.hstack(([iid], items))
            preds = pr.predict_next(sid, prev_iid, items_to_predict)

            preds[np.isnan(preds)] = 0
            preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties

            rank = (preds > preds[iid]).sum() + 1

            assert rank > 0
            if rank < cut_off:
                recall += 1
                mrr += 1.0/rank
            evalutation_point_count += 1

        prev_iid = iid

        count += 1

    print( 'END evaluation org in ', (time.perf_counter()-sc), 'c / ', (time.time()-st), 's' )
    print( '    HitRate ', recall/evalutation_point_count )
    print( '    MRR ', mrr/evalutation_point_count )

    return  recall/evalutation_point_count, mrr/evalutation_point_count
