import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import csv

#data config (all methods)
DATA_PATH = '../data/raw/'
DATA_PATH_PROCESSED = '../data/prepared/'
#DATA_FILE = 'yoochoose-clicks-10M'
DATA_FILE = 'yoochoose-clicks-1M'

#filtering config (all methods)
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5

#min date config
MIN_DATE = '2014-04-01'

#days test default config 
DAYS_TEST = 1

#slicing default config
NUM_SLICES = 10
DAYS_OFFSET = 0
DAYS_SHIFT = 5
DAYS_TRAIN = 9
DAYS_TEST = 1


#preprocessing from original gru4rec
def preprocess_org( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data_org( data, path_proc+file )

#preprocessing from original gru4rec but from a certain point in time
def preprocess_org_min_date( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, min_date=MIN_DATE ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    data = filter_min_date( data, min_date )
    split_data_org( data, path_proc+file )

#preprocessing adapted from original gru4rec
def preprocess_days_test( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data( data, path_proc+file, days_test )

#preprocessing to create data slices with a window
def preprocess_slices( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )
    
#just load and show info
def preprocess_info( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    
#just load and show info
def preprocess_buys( path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED ):
    
    data = load_data( path+file )
    data.to_csv( path_proc + file + '.txt', sep='\t', index=False)
    

    
def load_data( file ) : 
    
    #load csv
    data = pd.read_csv( file+'.dat', sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
    #specify header names
    data.columns = ['SessionId', 'TimeStr', 'ItemId']
    
    #convert time string to timestamp and remove the original column
    data['Time'] = data.TimeStr.apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
    del(data['TimeStr'])
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;


def filter_data( data, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ) : 
    
    #y?
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths >= min_session_length ].index)]
    
    #filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths >= min_session_length ].index)]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;

def filter_min_date( data, min_date='2014-04-01' ) :
    
    min_datetime = datetime.strptime(min_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
    
    #filter
    session_max_times = data.groupby('SessionId').Time.max()
    session_keep = session_max_times[ session_max_times > min_datetime.timestamp() ].index
    
    data = data[ np.in1d(data.SessionId, session_keep) ]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;



def split_data_org( data, output_file ) :
    
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_test = session_max_times[session_max_times >= tmax-86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
    
    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_valid = session_max_times[session_max_times >= tmax-86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv( output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
    valid.to_csv( output_file + '_train_valid.txt', sep='\t', index=False)


def dataStatistics(data, conf=None):
    sessionLenMap = {}
    print('dataStatistics start')  # same aEOS for all dbs
    bDataStatistics = False
    if (conf is not None):
        bDataStatistics = False
        try:
            bDataStatistics = conf['dataStatistics']
        except Exception:
            bDataStatistics = False
    else:
        bDataStatistics = True

    if (bDataStatistics):
        dataAsListOfLists = data.values.tolist()
        # look for variables locations
        indexOfSessionId = data.columns.get_loc("SessionId")
        session_length = 1
        firstEntry = dataAsListOfLists[0]
        currentSessionID = firstEntry[indexOfSessionId]
        entry_1 = firstEntry
        entry_2 = None
        i = 1
        totalSessions = 1
        dataLen = len(data)
        while i < len(data):
            if (i % 1000 == 0):
                print('processed dataStatistics' + str(i) + "/" + str(dataLen))

            entryList = dataAsListOfLists[i]
            entry = dataAsListOfLists[i]
            i += 1
            if (currentSessionID == entryList[indexOfSessionId] or currentSessionID == -1):
                # didn't moved to a new session
                currentSessionID = entryList[indexOfSessionId]
                entry_2 = entry_1
                entry_1 = entry
                session_length += 1
            else:
                # moved to a new session
                if (entry_2 is None or entry_1 is None):
                    print('unexpected less then 2 entries session')
                else:
                    # build new raw entry - based last two raws

                    if session_length in sessionLenMap:
                        num = sessionLenMap[session_length]
                        num += 1
                        sessionLenMap[session_length] = num
                    else:
                        sessionLenMap[session_length] = 1

                    totalSessions += 1

                    session_length = 1
                    currentSessionID = entry[indexOfSessionId]  # entry is a df in len  1
                    entry_1 = entry
                    entry_2 = None

        print('finished getting dataStatistics' + ' totalSessions[' + str(totalSessions) + ']data size[' + str(len(data)))
        print(str(data))
        print('new data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\n'.
              format(len(data), data.SessionId.nunique(), data.ItemId.nunique()))
    else:
        print('avoid getting dataStatistics')

    print('dataStatistics Done')
    return sessionLenMap, totalSessions,
    #########################################################




def split_data(data, output_file, days_test=DAYS_TEST, last_nth=None):
    return split_dataEx(data, output_file, 5, 2, days_test, last_nth)


def split_dataEx(data, output_file, minItemSupport, minSessionLength, days_test=DAYS_TEST, last_nth=None):

    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days_test)

    session_max_times = data.groupby('SessionId').Time.max()
    # Split train/test
    session_train = session_max_times[session_max_times < test_from.timestamp()].index
    session_test = session_max_times[session_max_times >= test_from.timestamp()].index
    train = data[np.in1d(data.SessionId, session_train)]

    if last_nth is not None:
        train.sort_values(['SessionId', 'Time'], inplace=True)
        session_data = list(data['SessionId'].values)
        lenth = int(len(session_data) / last_nth)
        session_data = session_data[-lenth:]
        for i in range(len(session_data)):
            if session_data[i] != session_data[i+1]:
                break
    
        train = data.reset_index()
        train = train[-lenth + i + 1:]
    
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)] # filter out from the test items that are not in the train
    tslength = test.groupby('SessionId').size()

    # Fix sessionLength Always = 2 # Remove sessions with less then X items
    #  (again) after splitting and removing items that does not appear on train ; missing removing items < 5 ...(bug)
    test = test[np.in1d(test.SessionId, tslength[tslength >= minSessionLength].index)]

    sessionLenMapTrain, TotalSessionsTrain, = dataStatistics(train)
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {} TotalSessionsTrain: {}'
          .format(len(train), train.SessionId.nunique(), train.ItemId.nunique(), TotalSessionsTrain))
    train.to_csv(output_file + (str(last_nth) if last_nth is not None else '') + '_train_full.txt', sep='\t', index=False)
    with open(output_file + '_trainSessionsLen.csv', 'w') as f:
        for key in sessionLenMapTrain.keys():
            f.write("%s,%s\n" % (key, sessionLenMapTrain[key]))

    sessionLenMapTest, TotalSessionsTest, = dataStatistics(test)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {} TotalSessionsTest: {}'
          .format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), TotalSessionsTest))
    test.to_csv(output_file + (str(last_nth) if last_nth is not None else '') + '_test.txt', sep='\t', index=False)
    with open(output_file + '_testSessionsLen.csv', 'w') as f:
        for key in sessionLenMapTest.keys():
            f.write("%s,%s\n" % (key, sessionLenMapTest[key]))
    
    data_end = datetime.fromtimestamp( train.Time.max(), timezone.utc )
    test_from = data_end - timedelta( days_test )
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[ session_max_times < test_from.timestamp() ].index
    session_valid = session_max_times[ session_max_times >= test_from.timestamp() ].index

    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv( output_file + (str(last_nth) if last_nth is not None else '') + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
    valid.to_csv( output_file + (str(last_nth) if last_nth is not None else '') + '_train_valid.txt', sep='\t', index=False)
    
    
def slice_data( data, output_file, num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT, days_train=DAYS_TRAIN, days_test=DAYS_TEST ): 
    
    for slice_id in range( 0, num_slices ) :
        split_data_slice( data, output_file, slice_id, days_offset+(slice_id*days_shift), days_train, days_test )

def split_data_slice( data, output_file, slice_id, days_offset, days_train, days_test ) :
    
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(), data_end.isoformat() ) )
    
    
    start = datetime.fromtimestamp( data.Time.min(), timezone.utc ) + timedelta( days_offset ) 
    middle =  start + timedelta( days_train )
    end =  middle + timedelta( days_test )
    
    #prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection( lower_end ))]
    
    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format( slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat(), end.date().isoformat() ) )
    
    #split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index
    
    train = data[np.in1d(data.SessionId, sessions_train)]
    
    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat() ) )
    
    train.to_csv(output_file + '_train_full.'+str(slice_id)+'.txt', sep='\t', index=False)
    
    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    
    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format( slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(), end.date().isoformat() ) )
    
    test.to_csv(output_file + '_test.'+str(slice_id)+'.txt', sep='\t', index=False)


# ------------------------------------- 
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    
    preprocess_slices();
