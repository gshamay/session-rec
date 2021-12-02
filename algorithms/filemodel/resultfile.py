import theano.misc.pkl_utils as pickle
import pandas as pd
import numpy as np
SEED = 42
np.random.seed(SEED)

class ResultFile:
    addOn = None
    '''
    FileModel( modelfile )
    Uses a trained algorithm, which was pickled to a file.

    Parameters
    -----------
    modelfile : string
        Path of the model to load

    '''

    def __init__(self, file, addOn =None):
        # config.experimental.unpickle_gpu_on_cpu = True
        self.file = file
        self.addOn = addOn
    
    def init(self, train, test=None, slice=None):
        file = self.file + ( ('.' + str(slice) + '.csv') if slice is not None else '' )
        if not '.csv' in file: 
            file = file + '.csv'
        self.recommendations = pd.read_csv( file, sep=';' )
                
        return
              
    def fit(self, train, test=None):
        
        self.pos = 0
        self.session_id = -1
        
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
        
        if session_id != self.session_id:
            self.pos = 0
            self.session_id = session_id
        
        recs = self.recommendations[(self.recommendations.SessionId == session_id) & (self.recommendations.Position == self.pos) ]
        if len(recs) == 0: 
            recs = self.recommendations[self.recommendations.SessionId == session_id]
            recs = recs.iloc[[self.pos]]
        items = recs.Recommendations.values[0]
        scores = recs.Scores.values[0]
        
        def convert( data, funct ):
            return map( lambda x: funct(x), data.split(',') )
        
        items = convert( items, int )
        scores = convert( scores, float )
        
        res = pd.Series( index=items, data=scores ) 
        
        self.pos += 1
        # todo: in case that the next res does not exists (train on EOS test on EOS)
        #  duplicated last prediction (it will be wrong anyway)

        if (self.addOn == 'random'):
            randValue = np.random()

            print('do random')

        if (self.addOn == 'naiveTrue'):

            print('do naiveTrue')

        
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
