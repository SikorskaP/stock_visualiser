import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as pdr
import pandas_ta as ta
import plotly.graph_objects as go
import ta

from plotly.subplots import make_subplots



class Functions():
    def __init__(self):
        pass

    if __name__ == '__main__':
        from options import Options
        from functions import Functions

  

    def display_dataframe():
        print(df)
        df.info()

    def moving_average(df):
        #df = Functions.ImportData()

        def wma(s, period):
            return s.rolling(period).apply(lambda x: ((np.arange(period)+1)*x).sum()/(np.arange(period)+1).sum(), raw=True)

        def hma(s, period):
            return WMA(WMA(s, period//2).multiply(2).sub(WMA(s, period)), int(np.sqrt(period)))

        def count_average(day):
            df['HMA_' + str(day) + '-day']=HMA(df['Close'],day)
            df['SMA_' + str(day) + '-day']=df['Close'].rolling(day).mean().shift() 

        def select_lengths():
            print("One of the many trading strategies utilising mocing averages puts two differing moving averages on a chart and highlights the "
            + "crossovers. When the shorter-term moving average crosses above the longer-term moving average, it's a buy signal. In the opposite situation, "
            + "it's a sell signal.")
            shortAverage = int(input("Select the first time inteval: \n> "))
            longAverage = int(input("Select the second time inteval: \n> "))
            if longAverage < shortAverage:
                dud = shortAverage
                shortAverage = longAverage
                longAverage = dud
            while (longAverage == shortAverage):
                longAverage = int(input("Wybierz inną liczbę. "))

            count_average(shortAverage)
            count_average(longAverage)
            return shortAverage, longAverage

        def plot():
            # strzałki w momencie krzyżowania się linii SMA
            df['SMA_signal'] = np.where(df['SMA_' + str(shortAverage) + '-day'] > df['SMA_' + str(longAverage) + '-day'], 1, 0)
            df['SMA_signal'] = np.where(df['SMA_' + str(shortAverage) + '-day'] < df['SMA_' + str(longAverage) + '-day'], -1, df['SMA_signal'])
            df.dropna(inplace=True)

            df['hold_return'] = np.log(df['Close']).diff()
            df['SMA_system_return'] = df['SMA_signal'] * df['hold_return']
            df['SMA_entry'] = df.SMA_signal.diff()

            # strzałki w momencie krzyżowania się linii HMA
            df['HMA_signal'] = np.where(df['HMA_' + str(shortAverage) + '-day'] > df['HMA_' + str(longAverage) + '-day'], 1, 0)
            df['HMA_signal'] = np.where(df['HMA_' + str(shortAverage) + '-day'] < df['HMA_' + str(longAverage) + '-day'], -1, df['HMA_signal'])
            df.dropna(inplace=True)

            df['HMA_return'] = np.log(df['Close']).diff()
            df['HMA_system_return'] = df['HMA_signal'] * df['HMA_return']
            df['HMA_entry'] = df.HMA_signal.diff()

            plt.rcParams['figure.figsize'] = 12, 6
            plt.grid(True, alpha = .3)
            plt.plot(df.iloc[-252:]['Close'], label = 'EURUSD')
            plt.plot(df.iloc[-252:]['HMA_' + str(shortAverage) + '-day'], label = 'HMA_' + str(shortAverage) + '-day', color='indigo')
            plt.plot(df.iloc[-252:]['HMA_' + str(longAverage) + '-day'], label = 'HMA_' + str(longAverage) + '-day', color='navy')

            plt.legend(loc=2);
            plt.show()

        def annotated_plot():
            plt.rcParams['figure.figsize'] = 12, 6
            plt.grid(True, alpha = .3)
            plt.plot(df.iloc[-252:]['Close'], label = 'EURUSD')
            plt.plot(df.iloc[-252:]['SMA_' + str(shortAverage) + '-day'], label = 'SMA_' + str(shortAverage) + '-day', color='indigo')
            plt.plot(df.iloc[-252:]['SMA_' + str(longAverage) + '-day'], label = 'SMA_' + str(longAverage) + '-day', color='navy')
            plt.plot(df[-252:].loc[df.SMA_entry == 2].index, df[-252:]['SMA_' + str(shortAverage) + '-day'][df.SMA_entry == 2], '^',
                    color = 'g', markersize = 12)
            plt.plot(df[-252:].loc[df.SMA_entry == -2].index, df[-252:]['SMA_' + str(longAverage) + '-day'][df.SMA_entry == -2], 'v',
                    color = 'r', markersize = 12)

            plt.plot(df.iloc[-252:]['HMA_' + str(shortAverage) + '-day'], label = 'HMA_' + str(shortAverage) + '-day', color='grey')
            plt.plot(df.iloc[-252:]['HMA_' + str(longAverage) + '-day'], label = 'HMA_' + str(longAverage) + '-day', color='darkorange')
            plt.plot(df[-252:].loc[df.HMA_entry == 2].index, df[-252:]['HMA_' + str(shortAverage) + '-day'][df.HMA_entry == 2], '^',
                    color = 'black', markersize = 12)
            plt.plot(df[-252:].loc[df.HMA_entry == -2].index, df[-252:]['HMA_' + str(longAverage) + '-day'][df.HMA_entry == -2], 'v',
                    color = 'purple', markersize = 12)

            plt.legend(loc=2);
            plt.show()
                
        shortAverage, longAverage = select_lengths()
        plot()
        annotated_plot()

    def stochastic_oscillator():
        def calculation(k_period,d_period):
            # max value of previous k periods
            df['n_high'] = df['High'].rolling(k_period).max()
            # min value of previous k periods
            df['n_low'] = df['Low'].rolling(k_period).min()
            # Uses the min/max values to calculate the %k (as a percentage)
            df['%K'] = (df['Close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])
            # Uses the %k to calculates a SMA over the past d values of %k
            df['%D'] = df['%K'].rolling(d_period).mean()

        def plot_stochastic():
                # Create our primary chart
                # the rows/cols arguments tell plotly we want two figures
                fig = make_subplots(rows=2, cols=1)  
                # Create our Candlestick chart with an overlaid price line
                fig.append_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        increasing_line_color='#ff9900',
                        decreasing_line_color='black',
                        showlegend=False
                    ), row=1, col=1  # <------------ upper chart
                )
                # price Line
                fig.append_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Open'],
                        line=dict(color='#ff9900', width=1),
                        name='Open',
                    ), row=1, col=1  # <------------ upper chart
                )
                # Fast Signal (%k)
                fig.append_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['STOCHk_' + str(k_period) + '_' + str(d_period) + '_3'],
                        line=dict(color='#ff9900', width=2),
                        name='Fast',
                    ), row=2, col=1  #  <------------ lower chart
                )
                # Slow signal (%d)
                fig.append_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['STOCHk_' + str(k_period) + '_' + str(d_period) + '_3'],
                        line=dict(color='#000000', width=2),
                        name='Slow'
                    ), row=2, col=1  #   <------------ lower chart
                )
                # Extend our y-axis a bit
                fig.update_yaxes(range=[-10, 110], row=2, col=1)
                # Add upper/lower bounds
                fig.add_hline(y=0, col=1, row=2, line_color="#666", line_width=2)
                fig.add_hline(y=100, col=1, row=2, line_color="#666", line_width=2)
                # Add overbought/oversold
                fig.add_hline(y=20, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
                fig.add_hline(y=80, col=1, row=2, line_color='#336699', line_width=2, line_dash='dash')
                # Make it pretty
                layout = go.Layout(
                    plot_bgcolor='#efefef',
                    # Font Families
                    font_family='Monospace',
                    font_color='#000000',
                    font_size=20,
                    xaxis=dict(
                        rangeslider=dict(
                            visible=False
                        )
                    )
                )
                fig.update_layout(layout)
                # View our chart in the system default HTML viewer (Chrome, Firefox, etc.)
                fig.show()

        k_period, d_period = Options.pick_periods()
        df.ta.stoch(high='High', low='Low', k=k_period, d=d_period, append=True)
        df.info()
        calculation(k_period, d_period)
        plot_stochastic()

    def double_stochastic():
        def count_double_stochastic():
            # Define periods
            k_period, d_period = Options.PickPeriods()

            # Adds a "n_high" column with max value of previous 14 periods
            df['n_high'] = df['High'].rolling(k_period).max()
            # Adds an "n_low" column with min value of previous 14 periods
            df['n_low'] = df['Low'].rolling(k_period).min()

            # Uses the min/max values to calculate the %k (as a percentage)
            df['fast%K'] = (df['Close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])
            # Uses the %k to calculates a SMA over the past 3 values of %k
            df.info()
            df['%K'] = (df['Close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])
            df['slowing%K'] = df['%K'].rolling(d_period).mean()
            df['f_high'] = df['slowing%K'].rolling(k_period).max()
            df['f_low'] = df['slowing%K'].rolling(k_period).min()

            df['double%K'] = (df['slowing%K'] - df['f_low']) * 100 / (df['f_high'] - df['f_low'])
            df['doubleSlowing%K'] = df['double%K'].rolling(d_period).mean()
            df['ds%d'] = df['doubleSlowing%K'].rolling(3).mean() # "3-period SMA of Double Slowing %K"

        def plot_double_stoch():
            plt.rcParams['figure.figsize'] = 12, 6
            fig, axs = plt.subplots(nrows=2, num="Double Stochastic")

            axs[0].plot(df['fast%K'], label='Fast %K')
            axs[1].plot(df['fast%K'], label='Fast %K')
            axs[1].plot(df['slowing%K'], label='Slowing %K', c='g')
            axs[1].plot(df['double%K'], label='Double %K', c='r')
            axs[1].plot(df['doubleSlowing%K'], label='Double Slowing %K', c='y')
            axs[1].plot(df['ds%d'], label='%D', c='b')
            plt.legend()
            plt.show()

        count_double_stochastic()
        plot_double_stoch()

    def bollinger_bands():
        def count_bollingerbands(prices, rate = 20):
            sma = prices.rolling(rate).mean() # <-- Get SMA for 20 days
            std = prices.rolling(rate).std() # <-- Get rolling standard deviation for 20 days
            bollinger_up = sma + std * 2 # Calculate top band
            bollinger_down = sma - std * 2 # Calculate bottom band
            return bollinger_up, bollinger_down

        def plot_bolbands():
            plt.rcParams['figure.figsize'] = 12, 6
            plt.subplots(num="Bollinger Bands")
            plt.title('Bollinger Bands')
            plt.xlabel('Days')
            plt.ylabel('Closing Prices')
            plt.plot(close, label='Closing Prices')
            plt.plot(bollinger_up, label='Bollinger Up', c='g')
            plt.plot(bollinger_down, label='Bollinger Down', c='r')
            plt.legend()
            plt.show()
        
        df.reset_index(inplace = True,drop = True)
        close = df['Close']
        bollinger_up, bollinger_down = count_bollingerbands(close)
        plot_bolbands()

    def rsi_macd():
        def count():
            df['RSI'] = ta.momentum.rsi(df.Close,window=14)
            df['MACD'] = ta.trend.macd_diff(df.Close)
            df.dropna(inplace=True)

        def plot():
            plt.rcParams['figure.figsize'] = 12, 6
            fig, axs = plt.subplots(nrows=2, num="MACD + RSI")

            axs[0].plot(df.iloc[-252:]['MACD'], label = 'MACD', color='navy')
            axs[1].plot(df.iloc[-252:]['RSI'], label = 'RSI', color='green')

            plt.legend(loc=2);
            plt.show()

        count()
        plot()

    def atr():
        def WWMA(values, n): # J. Welles Wilder's EMA 
            return values.ewm(alpha=1/n, min_periods=n, adjust=False).mean()

        def count_atr(n=14):
            data = df.copy()
            high = df['High']
            low = df['Low']
            close = df['Close']
            df['tr0'] = abs(high - low)
            df['tr1'] = abs(high - close.shift())
            df['tr2'] = abs(low - close.shift())
            tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            df['ATR'] = WWMA(tr, n)

        def plot_atr():
            plt.rcParams['figure.figsize'] = 12, 6
            candlestick = go.Candlestick(
                                x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                showlegend=False
                                )
            atrHigh = go.Scatter(x=df.index,
                            y=3*df["ATR"]+df["Close"],
                            name="High ATR"
                            )
            atrLow = go.Scatter(x=df.index,
                            y=-3*df["ATR"]+df["Close"],
                            name="Low ATR"
                            )
            fig = go.Figure(data=[candlestick, atrHigh, atrLow])

            fig.show()

        count_atr()
        plot_atr()
