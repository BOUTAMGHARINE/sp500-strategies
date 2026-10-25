from features_engineering import features_engineering


















def main():
    all_stocks_5yr = features_engineering()

    train = all_stocks_5yr[all_stocks_5yr["date"] < "2017-01-01"].copy()
    test = all_stocks_5yr[all_stocks_5yr["date"] >= "2017-01-01"].copy()

    

if __name__ == "__main__":
    main()