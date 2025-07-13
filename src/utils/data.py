import numpy as np
import pandas as pd
import pickle

YEARS = [2016, 2017]
QUARTERS = [1, 2, 3, 4]
QUARTER_DATES = {
    '2016_Q1': ['2012-10-01', '2015-10-01', '2016-01-04', '2016-04-01'],
    '2016_Q2': ['2013-01-02', '2016-01-04', '2016-04-01', '2016-07-01'],
    '2016_Q3': ['2013-04-02', '2016-04-01', '2016-07-01', '2016-10-03'],
    '2016_Q4': ['2013-07-02', '2016-07-01', '2016-10-03', '2017-01-03'],
    '2017_Q1': ['2013-10-01', '2016-10-03', '2017-01-03', '2017-04-03'],
    '2017_Q2': ['2014-01-02', '2017-01-03', '2017-04-03', '2017-07-03'],
    '2017_Q3': ['2014-04-01', '2017-04-03', '2017-07-03', '2017-10-02'],
    '2017_Q4': ['2014-07-01', '2017-07-03', '2017-10-02', '2018-01-02'],
}


def get_targets(year, Q, num=None):
    start_year = 2016

    with open('./data/company_buy_ticker.pickle', 'rb') as f:
        all_target = pickle.load(f)

        for t in all_target:

            if 'PYPL' in t:
                t.remove('PYPL')

            if 'KHC' in t:
                t.remove('KHC')

    if num == None:
        return all_target[year - start_year + Q - 1]

    else:
        return all_target[year - start_year + Q - 1][:num]


def define_dates(args, year=None, Q=None):
    pretrain_start = '2002-01-01'
    turbulance_start = '2012-01-02'

    if args.case == 3:
        key = str(year) + '_Q' + str(Q) 
        quarter_dates = get_quarter_dates()
        train_start, train_end, val_end, test_end = (date for date in quarter_dates[key])

    else:
        raise ValueError('Case not defined')
        
    if args.test:
        return val_end, test_end

    elif args.backtest:
        return train_start, train_end

    else:
        return pretrain_start, turbulance_start, train_start, train_end, val_end

    
def normalize(x):
    return (x - 1) * 100


def state_normalizer(state):
    state = state / state[:, -1:, -1:] # normalize to last close
    state = normalize(state) 
    return state


def get_init_action(dim, random=False, ew=False):
    if random:
        init_action = np.random.rand(dim)
        init_action /= np.sum(init_action)

    elif ew:
        init_action = np.ones(dim)
        init_action[0] = 0
        init_action /= np.sum(init_action)

    else:
        init_action = np.zeros(dim)
        init_action[0] = 1

    return init_action


def transform_data(args, history, dating):  
    data = history.copy()
    return history, dating, data


def generate_state(observation):
    cash = cash_state(observation)
    state = np.concatenate((cash, observation), 0)
    state = state_normalizer(state)
    return state


def cash_state(obs):
    cash = np.ones((1, obs.shape[1], obs.shape[2]))
    rfr = np.array([1.0])
    rate = np.exp(np.log(rfr) / 365)

    for i in range(cash.shape[1]-1):
        cash[:, i+1:, :] *= rate

    return cash


def get_quarter_dates():
    return QUARTER_DATES


def get_years_and_quarters():
    return YEARS, QUARTERS


def index_to_date(index, dating):
    return dating[index]    


def date_to_index(date_string, dating):
    date_idx = np.where(dating == date_string)
    return date_idx[0][0]  


def get_data_repo():
    repo = 'data/1617'
    return repo


def get_data(target_stocks, year, Q, status, bench=False):
    data_repo = get_data_repo()
    path = './' + data_repo + '/' + status + '/' + str(year) + 'Q' + str(Q)

    if bench:
        path += '_bench.csv'

    else:
        path += '.csv'

    df = pd.read_csv(path)
    df_hist = df[['date','open','high','low','close','volume','tic']]
    hist =[]
    
    for ticker in target_stocks:
        hist.append(np.expand_dims(df_hist[df_hist['tic']==ticker], axis=0))

    hist_np = np.concatenate(hist, axis=0)
    dating = hist_np[0][:,0]
    history = hist_np[:,:,1:5]
    num_training_time = len(dating)
    
    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[target_stocks.index(stock), :num_training_time, :]
    
    return target_history, dating