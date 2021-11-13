import time

'''
preprocessing method ["info","org","days_test","slice"]
    info: just load and show info
    org: from gru4rec (last day => test set)
    org_min_date: from gru4rec (last day => test set) but from a minimal date onwards
    days_test: adapted from gru4rec (last N days => test set)
    slice: new (create multiple train-test-combinations with a window approach  
    buys: load buys and safe file to prepared
'''
import sys
from pathlib import Path
import yaml
import importlib
import traceback
import os
import pandas as pd
import numpy as np

def main( conf ): 
    '''
    Execute experiments for the given configuration path
        --------
        conf: string
            Configuration path. Can be a single file or a folder.
        out: string
            Output folder path for endless run listening for new configurations. 
    '''
    print( 'Checking {}'.format( conf ) )
    
    file = Path( conf )
    if file.is_file():
        
        print( 'Loading file' )
        stream = open( str(file) )
        c = yaml.load(stream)
        stream.close()
        print( 'processing config ' + conf )
        
        try:
        
            run_file( c )
            print( 'finished config ' + conf )
            
        except (KeyboardInterrupt, SystemExit):
                        
            print( 'manually aborted config ' + conf )            
            raise
        
        except Exception:
            print( 'error for config ', file )
            traceback.print_exc()
            
        exit()
    
    print( 'File not found: ' + conf )
    
def run_file( conf ):
    
    #include preprocessing
    preprocessor = load_preprocessor( conf )
    
    #load data from raw and transform
    if 'sample_percentage' in conf['data']:
        data = preprocessor.load_data(conf['data']['folder'] + conf['data']['prefix'], sample_percentage=conf['data']['sample_percentage'])
    else:
        data = preprocessor.load_data( conf['data']['folder'] + conf['data']['prefix'] )
    if type(data) == tuple:
        extra = data[1:]
        data = data[0]
    # because in session-aware, pre-processing will be applied after data splitting
    if not(conf['mode'] == 'session_aware' and conf['type'] == 'window'):
        data = preprocessor.filter_data( data, **conf['filter'] )

    #todo: Add here aEOS ?
    #  diginetica Index(['SessionId', 'Time', 'ItemId', 'Date', 'Datestamp', 'TimeO', 'ItemSupport'],   dtype='object')

    # run on all data and add new aEOS
    session_length = 1
    firstEntry = data.iloc[[0]]
    newData = firstEntry
    currentSessionID = newData.iloc[0]['SessionId']
    entry_1 =  firstEntry
    entry_2 =  None
    i = 1


    while i < len(data):
        entry = data.iloc[[i]]
        currentIndex = i
        i+=1

        if(currentSessionID == data.iloc[currentIndex]['SessionId'] or currentSessionID == -1):
            #didn't moved to a new session
            currentSessionID = data.iloc[currentIndex]['SessionId']
            entry_2 = entry_1
            entry_1 = entry
            session_length+=1
            newData = newData.append(entry)
        else:
            #moved to a new session
            if(entry_2 is None or entry_1 is None):
                print('unexpected less then 2 entries session')
            else:
                # build new raw entry - based last two raws

                # todo: consider settingthe time according to the last two enries times
                #  timeBetweenLastTwoEnties = entry_1.iloc[0]['Time'] - entry_2.iloc[0]['Time']
                #  print('adding new line' + str(timeBetweenLastTwoEnties))

                newEntry  = entry_1.copy(deep=True)
                newEntry.ItemId = -1
                newEntry.Time = newEntry.Time + 1
                newData = newData.append(newEntry)
                newData = newData.append(entry)

                # add raw to the new Pos
                #data = pd.DataFrame(np.insert(data.values, i-1, values=newEntry, axis=0))
                #i+=1 #added a new row to the data - that we dont need to analyze

                #setting up new session data
                session_length = 1
                currentSessionID = entry.iloc[0]['SessionId']#entry is a df in len  1
                entry_1 = entry
                entry_2 = None


    ensure_dir( conf['output']['folder'] + conf['data']['prefix'] )
    #call method according to config
    if conf['type'] == 'single':
        preprocessor.split_data( data, conf['output']['folder'] + conf['data']['prefix'], **conf['params']  )
    elif conf['type'] == 'window':
        preprocessor.slice_data( data, conf['output']['folder'] + conf['data']['prefix'], **conf['params']  )
    elif conf['type'] == 'retrain':
        preprocessor.retrain_data(data, conf['output']['folder'] + conf['data']['prefix'], **conf['params'])
    else:
        if hasattr(preprocessor, conf['type']):
            method_to_call = getattr(preprocessor, conf['type'])
            method_to_call( data, conf['output']['folder'] + conf['data']['prefix'], **conf['params']  )
        else:
            print( 'preprocessing type not supported' )


###option...
def insert_row(idx, df, df_insert):
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]
    df = dfA.append(df_insert).append(dfB).reset_index(drop = True)
    return df

def load_preprocessor( conf ):
    '''
    Load the proprocessing module
        --------
        conf : conf
            Just the last part of the path, e.g., evaluation_last
    '''
    return importlib.import_module( 'preprocessing.'+conf['mode']+'.preprocess_' + conf['preprocessor'] )

def ensure_dir(file_path):
    '''
    Create all directories in the file_path if non-existent.
        --------
        file_path : string
            Path to the a file
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    '''
    Run the preprocessing configured above.
    '''
    
    if len( sys.argv ) == 2: 
        main( sys.argv[1] ) # for example: conf/preprocess/window/rsc15.yml
    else:
        print( 'Preprocessing configuration expected.' )
    
