import datetime
import yfinance as yf
from functions import Functions

class Options():   
    def get_positive_number(prompt):
        while True:
            try:
                num = int(input(prompt))
                if num <= 0:
                    raise ValueError("Number must be positive.")
                return num
            except ValueError:
                print("Invalid input. Please enter a positive number.")

    def print_menu():
        options = [
            "Display the dataframe",
            "Calculate the moving averages",
            "Calculate the Stochastic Oscillator",
            "Calculate the Double Stochastic",
            "Calculate the Bollinger Bands",
            "Calculate the RSI and the MACD",
            "Calculate the ATR (Candlestick Chart)",
            "Show the options menu",
            "Quit program"
        ]
        print("\nPlease choose from one of the following options...")
        for i, option in enumerate(options):
            print(f"{i+1}. {option}")

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
        match x:
            case '1':
                print("You have chosen to display the dataframe.")
                Functions.display_dataframe(df)
            case '2':
                print("You have chosen to calculate the moving averages.")
                Functions.moving_average(df)
            case '3':
                print("You have chosen to calculate the Stochastic Oscillator.")
                Functions.stochastic_oscillator(df)
            case '4':
                print("You have chosen to calculate the Double Stochastic.")
                Functions.double_stochastic(df)
            case '5':
                print("You have chosen to calculate the Bollinger Bands.")
                Functions.bollinger_bands(df)
            case '6':
                print("You have chosen to calculate the RSI and the MACD.")
                Functions.rsi_macd(df)
            case '7':
                print("You have chosen to calculate the ATR (Candlestick Chart).")
                Functions.atr(df)
            case 'o':
                print("You have chosen to show the options menu.")
            case _:
                return 0
    
    
    def pick_periods():
        k_period = int(input("\nPick the k line (default == 14P):\n> "))
        d_period = int(input("\nPick the d line (default == 3):\n> "))
        return k_period, d_period

    if __name__ == '__main__':
        from options import Options
        while True:
            Options.print_menu()
            optionSelected = get_positive_number("Please enter a positive number: \n> ")
            df = import_data()
            Options.choose_from_menu(optionSelected, df)
