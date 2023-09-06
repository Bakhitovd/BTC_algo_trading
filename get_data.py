import pandas as pd
from binance.futures import Futures as Client
from datetime import datetime, timedelta
import ta
import numpy as np

collumns = ['Open time','Open','High','Low','Close','Ignore','Close time','Ignore','Number of bisic data','Ignore','Ignore','Ignore']
time_mult = {'1m':60,'15m':60*15,'1h':60*60,'1d':60*60*24}


class get_new_data:
    def __init__(self, ticker, timeframe, limit, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        futures_client = Client()
        df=pd.DataFrame(futures_client.mark_price_klines(ticker, timeframe, **{"limit": limit}), columns = collumns)
        df['Open time'] = df.apply(lambda x: datetime.fromtimestamp(int(x['Open time'])/1000) , axis=1)
        df['Close time'] = df.apply(lambda x: datetime.fromtimestamp(int(x['Close time'])/1000) , axis=1)
        df.drop(columns = ['Ignore','Number of bisic data','Close time'], inplace = True)
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype('float32', copy=False)
        series = df[['Open time','Open']]
        series.columns = ['ds', 'y']
        self.df = df
        self.series = series

        
class get_hist_data:
    def __init__(self,start_date, end_date, ticker, timeframe, safe_df:bool=False):
        start_date = datetime.timestamp(datetime.fromisoformat(start_date))
        end_date = datetime.timestamp(datetime.fromisoformat(end_date)) 
        df = pd.DataFrame(columns = ['Open time', 'Open', 'High', 'Low', 'Close'])
        num_of_tab_1 = int((end_date - start_date)/time_mult[timeframe]//1000)
        futures_client = Client()
        for i in range(num_of_tab_1+1):
            cur_startTime = int(i*1000*time_mult[timeframe] + start_date)
            df1=pd.DataFrame(futures_client.klines(ticker, timeframe, **{"limit": 1000, 'startTime': cur_startTime*1000}), columns = collumns)
            df1['Open time'] = df1.apply(lambda x: datetime.fromtimestamp(int(x['Open time'])/1000) , axis=1)
            df1['Close time'] = df1.apply(lambda x: datetime.fromtimestamp(int(x['Close time'])/1000) , axis=1)
            df1.drop(columns = ['Ignore','Number of bisic data','Close time'], inplace = True)
            df1[['Open', 'High', 'Low', 'Close']] = df1[['Open', 'High', 'Low', 'Close']].astype('float32', copy=False)
            df = pd.concat([df, df1], ignore_index = True)
        if safe_df == True:
            start_for_safe = str(datetime.fromtimestamp(start_date).year) + "_" + str(datetime.fromtimestamp(start_date).month)+ "_" + str(datetime.fromtimestamp(start_date).day)
            stop_for_safe = str(datetime.fromtimestamp(end_date).year) + "_" + str(datetime.fromtimestamp(end_date).month)+ "_" + str(datetime.fromtimestamp(end_date).day)
            df.to_csv('df_BTC_{}_{}-{}.csv'.format(timeframe, start_for_safe, stop_for_safe), index = False)
        self.df = df

        
        
class indi_calc:
    def __init__(self, df):
        df['Open time']=pd.to_datetime(df['Open time'])
        self.df = df
        self.lenght = len(df)
        self.start_date =  df['Open time'][0]
        self.end_date = df['Open time'][self.lenght-1]
        self.time_step = df['Open time'][1].minute-df['Open time'][0].minute
    
    def RSI_ADX(self,RSI_window,ADX_window):
        rsi = ta.momentum.RSIIndicator(self.df['Open'], RSI_window, False)
        adx = ta.trend.ADXIndicator(self.df['High'], self.df['Low'], self.df['Close'] , ADX_window, False)
        df_rsi_adx = self.df.copy()
        df_rsi_adx['ADX'] = adx.adx()
        df_rsi_adx['-DI'] = adx.adx_neg()
        df_rsi_adx['+DI'] = adx.adx_pos()
        df_rsi_adx['RSI'] = rsi.rsi()
        return df_rsi_adx

class strategy_calc(indi_calc):
    
    def __init__(self, df, RSI_window, ADX_window):
        super().__init__(df)
        self.indi_df = super().RSI_ADX(RSI_window, ADX_window)
        self.arr = self.indi_df.iloc[:,1:].to_numpy().astype('int')

    def long(self, rsi_buy, adx_buy, SL:int=200, TP:int=200, time_step:int = 15):
        '''
        function for estimation RSI+ADX long positions 
        input:
        arr-numpy array witj shape = (num_steps, 8) columns - |Open | High | Low | Close | ADX | -DI | +DI | RSI |
        num_steps must be more than 100 + time_step
        return: np.array(['num_time':
                 ,'num_tp':
                 ,'num_stop':
                 ,'av_time':
                 ,'result':
                ])

        '''
        times = np.array([])
        num_stop = 0
        num_take = 0
        res = 0
        i = 20
        while i < len(self.arr)-time_step:
            if self.arr[i, 7] < rsi_buy and self.arr[i, 4] > adx_buy and self.arr[i, 6] > self.arr[i, 5]:
                if self.arr[i, 3] - SL > np.min(self.arr[i + 1 : 1 + i + time_step, 3]):
                    num_stop +=1
                    res += -SL
                elif self.arr[i, 3] + TP < np.max(self.arr[i + 1 : 1 + i + time_step, 3]):
                    num_take +=1
                    res += TP
                else:
                    times = np.append(times, [self.arr[i + time_step, 3] - self.arr[i, 3]])
                    res += self.arr[i + time_step, 3] - self.arr[i, 3]
                i += time_step
            else:
                i += 1
        if num_stop + num_take + len(times) == 0:
            return False

        return  np.array([int(len(times)), int(num_take), int(num_stop), np.mean(times).round(0), int(res)]).astype('int')
