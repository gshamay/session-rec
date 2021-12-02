import theano.misc.pkl_utils as pickle
import dill
import numpy as np
SEED = 42
np.random.seed(SEED)

class FileModel:
    addOn = None
    '''
    FileModel( modelfile )
    Uses a trained algorithm, which was pickled to a file.

    Parameters
    -----------
    modelfile : string
        Path of the model to load

    '''

    def __init__(self, modelfile,addOn =None):
        # config.experimental.unpickle_gpu_on_cpu = True
        self.model = dill.load(open(modelfile, 'rb'))
        self.addOn = addOn

    #fix Error: AttributeError: 'FileModel' object has no attribute 'clear'


    def clear(self):
        print("Clear in FileModfel - call model clear")
        self.model.clear()
        return

    def fit(self, train, test=None):
        print("fit in FileModfel - do nothing")
        return

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
        retVal = self.model.predict_next(session_id, input_item_id, predict_for_item_ids, skip, mode_type)
        if (self.addOn != None):
            # in case that some prediction was not a valid number (NaN) -it's probability is zeroed
            retVal[np.isnan(retVal)] = 0
            aEOSItemId = -1
            retVal.sort_values(ascending=False, inplace=True)  # sort preds according to the predicted probability
            if (self.addOn == 'random'):
                print('do random')
                randRate = np.random.random()
                highestValue = retVal[retVal.index[0]]
                randValue = randRate * highestValue
                retVal[aEOSItemId] = randValue

            if (self.addOn == 'naiveTrue'):
                print('do naiveTrue')
                defaultKLocation = 5
                value4 = retVal[retVal.index[defaultKLocation - 1]]
                value5 = retVal[retVal.index[defaultKLocation]]
                newValue5 = (value4 + value5) / 2
                if (newValue5 == 0):
                    defaultValueToSetInResults = 0.01
                    newValue5
                    defaultValueToSetInResults

                retVal[aEOSItemId] = newValue5

        return retVal


    def predict_next_batch(self, session_ids, input_item_ids, predict_for_item_ids=None, batch=100):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        retVal = self.model.predict_next_batch(session_ids, input_item_ids, predict_for_item_ids, batch)
        if (self.addOn == 'random'):
            print('do random')

        if (self.addOn == 'naiveTrue'):
            print('do naiveTrue')

        return retVal


