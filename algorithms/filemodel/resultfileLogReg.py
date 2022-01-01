import theano.misc.pkl_utils as pickle
import pandas as pd
import numpy as np
import time

SEED = 42
np.random.seed(SEED)


class resultfileLogReg:
    addOn = None
    file2 = None
    recommendations2 = None
    '''
    resultfileLogReg( modelfile )
    Uses a trained algorithm, which was pickled to a file.

    Parameters
    -----------
    modelfile : string
        Path of the model to load

    '''

    def __init__(self, file, addOn=None, file2=None, ):
        # config.experimental.unpickle_gpu_on_cpu = True
        self.file = file
        self.file2 = file2
        self.addOn = addOn

    def init(self, train, test=None, slice=None):
        file = self.file + (('.' + str(slice) + '.csv') if slice is not None else '')
        if not '.csv' in file:
            file = file + '.csv'
        self.recommendations = pd.read_csv(file, sep=';')

        if (self.file2 is not None):
            file2 = self.file2 + (('.' + str(slice) + '.csv') if slice is not None else '')
            if not '.csv' in file2:
                file2 = file2 + '.csv'
            self.recommendations2 = pd.read_csv(file2, sep=';')

        return


    def fit(self, train, test=None,session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.pos = 0
        self.session_id = -1

        ###############################
        # train the LR model / Begin
        sc = time.clock()
        st = time.time()
        time_sum = 0
        time_sum_clock = 0
        time_count = 0
        count = 0
        train.sort_values([session_key, time_key], inplace=True)
        items_to_predict = train[item_key].unique()
        prev_iid, prev_sid = -1, -1
        pos = 0
        actions = len(train)

        for i in range(len(train)):
            if count % 1000 == 0:
                print('    eval process: ', count, ' of ', actions, ' actions: ', (count / actions * 100.0), ' % in',
                      (time.time() - st), 's')

            iid = train[item_key].values[i] # the actual Item ID
            isEOS = (iid <= aEOSBaseIDValue)

            sid = train[session_key].values[i]
            ts = train[time_key].values[i]
            if prev_sid != sid:
                prev_sid = sid
                pos = 0
                # there is no seesion in len == 1 therefore there os no need to check EOS here ;
                # todo: need to add it to the LR ? (seesionLen=1 --> no ?  )
            else:
                crs = time.clock()
                trs = time.time()

                # get the prediction from the model / file results
                preds = self.predict_next(sid, prev_iid, items_to_predict, timestamp=ts)  # predict all sub sessions
                # preds contain now a list of all possible items with their probabilities to be the next item
                # todo : Replace here self. to predict model to use with none file model

                # refine the predictions
                preds[np.isnan(preds)] = 0
                # in case that some prediction was not a valid number (NaN) -it's probability is zeroed
                # preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties # todo: ?
                preds.sort_values(ascending=False,inplace=True)
                # sort preds according to the predicted probability

                ############################################################
                # Look for aEOS predictions --> take it's max score
                maxUsedK = len(preds) # 50 # todo: consider limit this to top K
                aEOSBaseIDValue = -1
                foundAEOS = False
                aEOSMaxPredictedValue = 0
                #defaultMinValueToPushDownPrediction = -0.01
                for i in range(maxUsedK):
                    iKey = preds.index[i]
                    if (iKey <= aEOSBaseIDValue):
                        if (not foundAEOS):
                            foundAEOS = True
                            aEOSMaxPredictedValue = preds[iKey]
                            break

                ############################################################
                # train the LR model
                sessionLen = pos + 1
                # print(str(aEOSMaxPredictedValue) + str(sessionLen) + str(isEOS) )
                ############################################################
                time_sum_clock += time.clock() - crs
                time_sum += time.time() - trs
                time_count += 1
                pos += 1

            prev_iid = iid #
            count += 1 # position in the train set

        print('END train LR in ', (time.clock() - sc), 'c / ', (time.time() - st), 's')
        print('    avg rt ', (time_sum / time_count), 's / ', (time_sum_clock / time_count), 'c')
        print('    time count ', (time_count), 'count/', (time_sum), ' sum')
        # train the LR model / End
        ###############################


        return



    missingPredicitons = 0



    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, mode_type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''

        if session_id != self.session_id:
            self.pos = 0
            self.session_id = session_id

        recs = self.recommendations[
            (self.recommendations.SessionId == session_id) & (self.recommendations.Position == self.pos)]

        # Work on exiting results, model that was trained and tested w/o aEOS, but now with aEOS in the expected output
        #  This provides the ability to test an exiting model with aEOS random variation ( when addOn is applied )
        # in case that the next res does not exists (train on EOS test on EOS)
        #  duplicated last prediction (it will be wrong anyway)
        if (len(recs) == 0):
            recs = self.recommendations[self.recommendations.SessionId == session_id]
            if (self.pos >= len(recs)):
                self.pos = self.pos - 1
                recs = self.recommendations[
                    (self.recommendations.SessionId == session_id) & (self.recommendations.Position == (self.pos))]

            else:
                recs = recs.iloc[[self.pos]]

        if (len(recs) == 0):
            self.missingPredicitons += 1
            recs = self.recommendations.iloc[[self.pos]]
            print('missing predictions session[' + str(session_id) + ']pos[' + str(
                self.pos) + ']missingPredicitons[' + str(self.missingPredicitons))

        try:
            items = recs.Recommendations.values[0]
            scores = recs.Scores.values[0]
        except IndexError:
            print("error! we should not get here!!!")

        def convert(data, funct):
            return map(lambda x: funct(x), data.split(','))

        items = convert(items, int)
        scores = convert(scores, float)

        res = pd.Series(index=items, data=scores)

        self.pos += 1

        # todo: refactor: identical code to fileModel
        if (self.addOn != None):
            # in case that some prediction was not a valid number (NaN) -it's probability is zeroed
            res[np.isnan(res)] = 0
            aEOSItemId = -1
            res.sort_values(ascending=False, inplace=True)  # sort preds according to the predicted probability
            if (self.addOn == 'random'):
                # print('do random')
                randRate = np.random.random()
                highestValue = res[res.index[0]]
                randValue = randRate * highestValue
                res[aEOSItemId] = randValue

            if (self.addOn == 'naiveTrue'):
                # print('do naiveTrue')
                defaultKLocation = 5
                value4 = res[res.index[defaultKLocation - 1]]
                value5 = res[res.index[defaultKLocation]]
                newValue5 = (value4 + value5) / 2
                if (newValue5 == 0):
                    defaultValueToSetInResults = 0.01
                    newValue5
                    defaultValueToSetInResults

                res[aEOSItemId] = newValue5

            # if (self.addOn == '2Files'):
            #     recs = self.recommendations2[(self.recommendations2.SessionId == session_id)]
            #     for rec in recs:
            #         items = rec.Recommendations.values[0]
            #         scores = rec.Scores.values[0]
            #         if(aEOSItemId in items)
            #
            #     # print('do random')
            #     highestValue = res[res.index[0]]
            #     randValue = randRate * highestValue
            #     res[aEOSItemId] = randValue

        return res

    def clear(self):
        del self.recommendations

    def support_users(self):
        '''
          whether it is a session-based or session-aware algorithm
          (if returns True, method "predict_with_training_data" must be defined as well)

          Parameters
          --------

          Returns
          --------
          True : if it is session-aware
          False : if it is session-based
        '''
        return False
