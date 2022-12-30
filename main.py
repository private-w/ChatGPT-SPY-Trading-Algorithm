import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yfinance as yf
import requests



def trade_spy(data, clf):
  # initialize variables
  balance = 100000
  shares_owned = 0
  buy_price = 0
  sell_price = 0

  for index, row in data.iterrows():
    # get the prediction from the model
    prediction = clf.predict([[row['Open'], row['High'], row['Low'], row['Close']]])[0]

    # if the model predicts the price will go up and we don't own any shares, try to buy
    if prediction == 'up' and shares_owned == 0:
      if row['Close'] < buy_price or buy_price == 0:
        buy_price = row['Close']
        shares_owned = balance / buy_price
        balance = 0

    # if the model predicts the price will go down and we own shares, try to sell
    elif prediction == 'down' and shares_owned > 0:
      if row['Close'] > sell_price or sell_price == 0:
        sell_price = row['Close']
        balance = sell_price * shares_owned
        shares_owned = 0
        buy_price = 0
        sell_price = 0

  # calculate profit
  profit = balance - 100000

  return profit



def get_usa_spending_data():
  url = "https://api.usaspending.gov"

  endpoint= "/api/v2/references/toptier_agencies/"
  response = requests.get(f"{url}{endpoint}")
  data = response.json()
  toptier_agencies_df = pd.DataFrame(data["results"])
  # print(toptier_agencies_df["agency_id"], toptier_agencies_df["agency_name"])
  toptier_agencies_list = toptier_agencies_df["agency_id"].to_list()
  # print(toptier_agencies_list)

  all_spending_df = pd.DataFrame()
  for toptier_agencies_id in toptier_agencies_list:
    endpoint = "/api/v2/award_spending/recipient"
    page = 1
    LIMIT = 500

    results_df = pd.DataFrame()
    query_more_pages = True
    while(query_more_pages):
      payload = {"awarding_agency_id": toptier_agencies_id, "fiscal_year": 2021, "limit": LIMIT, "page": page}
      

      response = requests.get(f"{url}{endpoint}", params=payload)
      if response.status_code == 200:
        try:
          data = response.json()
          df = pd.DataFrame(data["results"])
          if df.shape[0] == 0:
            query_more_pages = False
          
          results_df = pd.concat([results_df, df], ignore_index=True)
          page += 1
        except KeyError:
          query_more_pages = False
        except:
          query_more_pages = False
      else:
        query_more_pages = False

    print(results_df)

    all_spending_df = pd.concat([all_spending_df, results_df], ignore_index=True)
    
  # print(all_spending_df)
  all_spending_df.to_csv("all_spending.csv", index=False)

  return all_spending_df


from os.path import exists

file_exists = exists("all_spending.csv")
print(file_exists)

if file_exists:
  all_spending_df = pd.read_csv("all_spending.csv")
else:
  all_spending_df = get_usa_spending_data()


# # get the stock data for SPY
# spy = yf.Ticker("SPY")

# # get the current date and time
# now = datetime.datetime.now()

# # calculate the start date as 3 years ago
# start_date = now - datetime.timedelta(days=365*3)

# # calculate the end date as 3 months ago
# end_date = now - datetime.timedelta(days=90)

# # retrieve the historical price data for the past 2 years, excluding the most recent 3 months
# data = spy.history(start=start_date, end=end_date)

# # create a new column with the label 'up' if the price increased from the previous day, 'down' if it decreased, and 'flat' if it remained the same
# data['label'] = np.where(data['Close'].shift(-1) > data['Close'], 'up',
#                          np.where(data['Close'].shift(-1) < data['Close'], 'down', 'flat'))


# # create a feature matrix with the 'Open', 'High', 'Low', and 'Close' columns
# X = pd.DataFrame(data[['Open', 'High', 'Low', 'Close']], columns=['Open', 'High', 'Low', 'Close'])

# # create a target vector with the 'label' column
# y = data['label']


# # split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # create a random forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)

# # fit the classifier to the training data
# clf.fit(X_train, y_train)

# # evaluate the classifier on the test data
# accuracy = clf.score(X_test, y_test)

# # run the trading algorithm
# profit = trade_spy(data, clf)

# # print the profit
# print(profit)
# print(f'Test accuracy: {accuracy:.2f}')
