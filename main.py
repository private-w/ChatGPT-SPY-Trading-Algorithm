import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yfinance as yf


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


# get the stock data for SPY
spy = yf.Ticker("SPY")

# get the current date and time
now = datetime.datetime.now()

# calculate the start date as 3 years ago
start_date = now - datetime.timedelta(days=365*3)

# calculate the end date as 3 months ago
end_date = now - datetime.timedelta(days=90)

# retrieve the historical price data for the past 2 years, excluding the most recent 3 months
data = spy.history(start=start_date, end=end_date)

# create a new column with the label 'up' if the price increased from the previous day, 'down' if it decreased, and 'flat' if it remained the same
data['label'] = np.where(data['Close'].shift(-1) > data['Close'], 'up',
                         np.where(data['Close'].shift(-1) < data['Close'], 'down', 'flat'))


# create a feature matrix with the 'Open', 'High', 'Low', and 'Close' columns
X = pd.DataFrame(data[['Open', 'High', 'Low', 'Close']], columns=['Open', 'High', 'Low', 'Close'])

# create a target vector with the 'label' column
y = data['label']


# split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# fit the classifier to the training data
clf.fit(X_train, y_train)

# evaluate the classifier on the test data
accuracy = clf.score(X_test, y_test)

# run the trading algorithm
profit = trade_spy(data, clf)

# print the profit
print(profit)
print(f'Test accuracy: {accuracy:.2f}')
