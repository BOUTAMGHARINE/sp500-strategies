
import pandas as pd
import numpy as np








def data_preprocessing():

    all_stocks_5yr = pd.read_csv("../data/all_stocks_5yr.csv")
    Historical_prices = pd.read_csv("../data/HistoricalPrices.csv")

    all_stocks_5yr.columns = all_stocks_5yr.columns.str.strip()
    Historical_prices.columns = Historical_prices.columns.str.strip()

    print(all_stocks_5yr.isnull().sum())
    print(Historical_prices.isnull().sum())
    
    print("\n--- all_stocks_5yr ---")
    print(all_stocks_5yr.head())
    print(all_stocks_5yr.info())

    print("\n--- HistoricalPrices ---")
    print(Historical_prices.head())
    print(Historical_prices.info())
    
    missing_rows = all_stocks_5yr[
        all_stocks_5yr.isnull().any(axis=1)
    ]

    print("\n--- Missing rows ---")
    print(missing_rows)

    print("\n--- Number of missing values per company ---")
    print(
        all_stocks_5yr.groupby("Name")[["open", "high", "low"]]
        .apply(lambda df: df.isnull().sum())
        .loc[lambda df: df.sum(axis=1) > 0]
    )
    
    all_stocks_5yr = all_stocks_5yr.dropna()
    
    print("*" * 30)
    
    print(all_stocks_5yr.isnull().sum())
    print(Historical_prices.isnull().sum())
    
    all_stocks_5yr["date"] = pd.to_datetime(all_stocks_5yr["date"])
    Historical_prices["Date"] = pd.to_datetime(Historical_prices["Date"],format="%m/%d/%y")
    all_stocks_5yr = all_stocks_5yr.sort_values(["Name","date"]).reset_index(drop=True)
    Historical_prices = Historical_prices.sort_values("Date").reset_index( drop = True )
    
    return all_stocks_5yr , Historical_prices    
    
    
    
    
    


    

all_stocks_5yr , Historical_prices = data_preprocessing()

print(all_stocks_5yr.dtypes)
print(Historical_prices.dtypes)


