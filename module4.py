import numpy as np
import pandas as pd
import math
import talib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import ParamterGrid
from pprint import pprint
from sklearn.preprocessing import scale

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import tensowflow as tf
from keras.layers import Dense, Dropout

stocks = ['AMD', 'CHK', 'QQQ']
full_df = pd.concat([amd_df, chk_df, qqq_df], axis = 1).dropna()
full_df.head()

# calculate daily returns of stocks
returns_daily = full_df.pct_change()

# resample the full dataframe to monthly timeframe
monthly_df = full_df.resample('BMS').first()

# calculate monthly returns of the stocks
returns_monthly = monthly_df.pct_change().dropna()
print(returns_monthly.tail())

covariances = {}

for i in returns_monthly.index:
	rtd_idx = returns_daily.index
	# mask daily returns for each month (and year) & calculate covariance
	mask = (rtd_idx.month == i.month) & (rtd_idx.year == i.year)
	covariances[i] = returns_daily[mask].cov()

print(covariances[i])

for date in covariances.keys():
	cov = covariances[date]
	for single_portfolio in range(5000):
		weights = np.random.random(3)
		weights /= np.sum(weights)
		returns = np.dot(weights,returns_monthly.loc[date])
		volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
		portfolio_volatility.setdefault(date,[]).append(returns)
		portfolio_volatility.setdefault(date,[]).append(volatility)
		portfolio_weights.setdefault(date,[]).append(weights)

date = sorted(covariances.keys())[-1]
# plot efficient frontier
plt.scatter(x = portfolio_volatility[date], 
			y = portfolio_returns[date],
			alpha = 0.5)
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()


###########################

# empty dictionaries for sharpe ratios and best sharpe indexes by date
sharpe_ratio, max_shar_idxs = {}, {}
# loop through dates and get sharpe ratio for each portfolio
for date in portfolio_returns.keys():
	for i, ret in enumerate(portfolio_returns[date]):
		volatility = portfolio_volatility[date][i]
		sharp_ratio.setdefault(date,[]).append(ret/volatility)
	# get the index of the best sharpe ratio for each date
	max_sharpe_idxs[date] = np.argmax(sharpe_ratio[date])

	#calculate exponentially-weighted moving average of daily returns
	ewma_daily = returns_daily.ewm(span = 30).mean()

	#resample daily returns to first business day of the month
	ewma_monthly = ewma_daily.resample('BMS').first()

	# shift ewma 1 month forward
	ewma_monhly = ewma_monthly.shift(1).dropna()

targets, features = [], []

# create features from price history and targets as ideal portfolio
for date, ewma in ewma_monthly.iterrows():
	# get the index of the best sharpe ratio
	best_idx = max_sharpe_idxs[date]
	targets.append(portfolio_weights[date][best_idx])
	features.append(ewma)

targets = np.array(targets)
features = np.array(features)

#latest date

date = sorted(covariances.keys())[-1]
cur_returns = portfolio_returns[date]
cur_volatility = portfolio_volatility[date]
plt.scatter(x = cur_volatility,
	y = cur_returns, 
	alpha = 0.1,
	color = 'blue')

best_idx = max_sharpe_idxs[date]

plt.scatter(cur_volatility[best_idx],
	cur_returns[best_idx],
	marker = 'x',
	color = 'orange')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()


###########################


# make train and test features
train_size = int(0.8*features.shape[0])
train_features = features[:train_size]
train_targets = targets[:train_size]
test_features = features[train_size:]
test_targets = targets[train_size:]

print(features.shape)

# fit the model and check scores on train and test

rfr = RandomForestRegressor(n_estimators = 300, random_state = 42)
rfr.fit(train_features, train_targets)
print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))

# get predictions from model on train and test
test_predictions = rfr.predict(test_features)
# calculate and plot returns from our RF predictions and the qqq returns
test_returns = np.sum(returns_monthly.iloc[train_size:]*test_predictions, axis = 1)
plt.plot(test_returns, label ='algo')
plt.plot(returns_monthly['QQQ'].iloc[train_size:], label ='QQQ')
plt.legend()
plt.show()

cash = 1000
algo_cash = [cash]
for r in test_returns:
	cash *= 1+r
	algo_cash.append(cash)

#calculate performance for QQQ

cash = 1000 # reset cash amount
qqq_cash = [cash]
for r in returns_monthly['QQQ'].iloc[train_size:]:
	cash*=1 + r
	qqq_cash.append(cash)

print('algo returns:', (algo_cash[-1] - algo_cash[0])/algo_cash[0])
print('QQQ returns:', (qqq_cash[-1] - qqq_cash[0])/qqq_cash[0])

plt.plot(algo_cash, label = 'algo')
plt.plot(qqq_cash, label = 'QQQ')
plt.ylabel('$')
plt.legend() # show the legend
plt.show()


###########################################

# Join 3 stock dataframes together
full_df = pd.concat([lng_df, spy_df, smlv_df], axis=1).dropna()

# Resample the full dataframe to monthly timeframe
monthly_df = full_df.resample('BMS').first()

# Calculate daily returns of stocks
returns_daily = full_df.pct_change()

# Calculate monthly returns of the stocks
returns_monthly = monthly_df.pct_change().dropna()
print(returns_monthly.tail())

# Daily covariance of stocks (for each monthly period)
covariances = {}
rtd_idx = returns_daily.index
for i in returns_monthly.index:    
    # Mask daily returns for each month and year, and calculate covariance
    mask = (rtd_idx.month == i.month) & (rtd_idx.year == i.year)
    
    # Use the mask to get daily returns for the current month and year of monthy returns index
    covariances[i] = returns_daily[mask].cov()

print(covariances[i])

portfolio_returns, portfolio_volatility, portfolio_weights = {}, {}, {}

# Get portfolio performances at each month
for date in sorted(covariances.keys()):
    cov = covariances[date]
    for portfolio in range(10):
        weights = np.random.random(3)
        weights /= np.sum(weights) # /= divides weights by their sum to normalize
        returns = np.dot(weights.T, returns_monthly.loc[date])
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        portfolio_returns.setdefault(date, []).append(returns)
        portfolio_volatility.setdefault(date, []).append(volatility)
        portfolio_weights.setdefault(date, []).append(weights)
        
print(portfolio_weights[date][0])

# Get latest date of available data
date = sorted(covariances.keys())[-1]  

# Plot efficient frontier
# warning: this can take at least 10s for the plot to execute...
plt.scatter(x=portfolio_volatility[date], y=portfolio_returns[date],  alpha=0.1)
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()

# Empty dictionaries for sharpe ratios and best sharpe indexes by date
sharpe_ratio, max_sharpe_idxs = {}, {}

# Loop through dates and get sharpe ratio for each portfolio
for date in portfolio_returns.keys():
    for i, ret in enumerate(portfolio_returns[date]):
    
        # Divide returns by the volatility for the date and index, i
        sharpe_ratio.setdefault(date, []).append(ret / portfolio_volatility[date][i])

    # Get the index of the best sharpe ratio for each date
    max_sharpe_idxs[date] = np.argmax(sharpe_ratio[date])

print(portfolio_returns[date][max_sharpe_idxs[date]])

# Calculate exponentially-weighted moving average of daily returns
ewma_daily = returns_daily.ewm(span=30).mean()

# Resample daily returns to first business day of the month with the first day for that month
ewma_monthly = ewma_daily.resample('BMS').first()

# Shift ewma for the month by 1 month forward so we can use it as a feature for future predictions 
ewma_monthly = ewma_monthly.shift(1).dropna()

print(ewma_monthly.iloc[-1])

targets, features = [], []

# Create features from price history and targets as ideal portfolio
for date, ewma in ewma_monthly.iterrows():

    # Get the index of the best sharpe ratio
    best_idx = max_sharpe_idxs[date]
    targets.append(portfolio_weights[date][best_idx])
    features.append(ewma)  # add ewma to features

targets = np.array(targets)
features = np.array(features)
print(targets[-5:])

# Get most recent (current) returns and volatility
date = sorted(covariances.keys())[-1]
cur_returns = portfolio_returns[date]
cur_volatility = portfolio_volatility[date]

# Plot efficient frontier with sharpe as point
plt.scatter(x=cur_volatility, y=cur_returns, alpha=0.1, color='blue')
best_idx = max_sharpe_idxs[date]

# Place an orange "X" on the point with the best Sharpe ratio
plt.scatter(x=cur_volatility[best_idx], y=cur_returns[best_idx], marker='x', color='orange')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()

# Get predictions from model on train and test
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)

# Calculate and plot returns from our RF predictions and the SPY returns
test_returns = np.sum(returns_monthly.iloc[train_size:] * test_predictions, axis=1)
plt.plot(test_returns, label='algo')
plt.plot(returns_monthly['SPY'].iloc[train_size:], label='SPY')
plt.legend()
plt.show()

# Calculate the effect of our portfolio selection on a hypothetical $1k investment
cash = 1000
algo_cash, spy_cash = [cash], [cash]  # set equal starting cash amounts
for r in test_returns:
    cash *= 1 + r
    algo_cash.append(cash)

# Calculate performance for SPY
cash = 1000  # reset cash amount
for r in returns_monthly['SPY'].iloc[train_size:]:
    cash *= 1 + r
    spy_cash.append(cash)

print('algo returns:', (algo_cash[-1] - algo_cash[0]) / algo_cash[0])
print('SPY returns:', (spy_cash[-1] - spy_cash[0]) / spy_cash[0])

# Plot the algo_cash and spy_cash to compare overall returns
plt.plot(algo_cash,label = 'algo')
plt.plot(spy_cash, label='SPY')
plt.legend()  # show the legend
plt.show()