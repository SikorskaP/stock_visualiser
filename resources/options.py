import datetime
import yfinance as yf
import pandas as pd
from resources.functions import Functions

class Options():   
    @staticmethod
    def get_positive_number(prompt):
        while True:
            try:
                num = int(input(prompt))
                if num <= 0:
                    raise ValueError("Number must be positive.")
                return num
            except ValueError:
                print("Invalid input. Please enter a positive number.")
    
    @staticmethod
    def print_menu(options):
        print("\nPlease choose from one of the following options...")
        for i, option in enumerate(options):
            print(f"{i+1}. {option}")
    
    def main_menu():
        options = [
            "Display the dataframe",  # OR CHANGE DOWNLOAD PERIOD
            "Calculate the moving averages",
            "Calculate the Stochastic Oscillator",
            "Calculate the Double Stochastic",
            "Calculate the Bollinger Bands",
            "Calculate the RSI and the MACD",
            "Calculate the ATR (Candlestick Chart)",
            "Show the options menu",
            "Quit program"
        ]
        Options.print_menu(options)
    
    def options_menu(self):
        print("You have chosen to display the options menu.")
        options = [
            "Change import length of dataframe",  
            "Exit options and go back to the program"
        ]
        Options.print_menu(options)
        selection = Options.get_positive_number("Please pick the function you want to execute: \n> ")
        match selection:
            case 1:
                print("You have chosen to change import length of dataframe.")
                df = Options.import_data()
                print(df)
                return df
            case 2:
                pass

    @staticmethod
    def import_data(): 
        from options import Options        
        okres = Options.get_positive_number("\nHow many months do you want to download the data from? \n> ")
        BeginDay = datetime.date.today() - datetime.timedelta(days = 31 * okres)
        EndDay = datetime.date.today()

        df = yf.download('EURUSD=X', BeginDay,EndDay)
        df.drop(columns=['Adj Close'], inplace = True)
        print(1, df)
        return df   
    
    def choose_from_menu(x, df):
        options = {
            1: Functions.display_dataframe,
            2: Functions.moving_average,
            3: Functions.stochastic_oscillator,
            4: Functions.double_stochastic,
            5: Functions.bollinger_bands,
            6: Functions.rsi_macd,
            7: Functions.atr,
            8: Options.options_menu,
            9: quit,
        }
        
        func = options.get(x, lambda: print("Invalid input. Please enter a number between 1 and 9."))
        func(df)
        return df

    if __name__ == '__main__':
        from options import Options
        df = Options.import_data()
        
        while True:
            Options.main_menu()
            option_selected = Options.get_positive_number("Please pick the function you want to execute: \n> ")
            df = Options.choose_from_menu(option_selected, df)
