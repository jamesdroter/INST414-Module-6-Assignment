import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('4_tools.csv')

#Handle missing data and create target variable
data['outs_above_average'].fillna(data['outs_above_average'].mean(), inplace=True)
data['CPS'] = data['slg_percent'] * 0.4 + data['on_base_percent'] * 0.3 + data['outs_above_average'] * 0.2 + data['sprint_speed'] * 0.1

#Select features and target
X = data[['slg_percent', 'on_base_percent', 'outs_above_average', 'sprint_speed']]
y = data['CPS']

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'Mean Absolute Error: {mae}')
print(f'R²: {r2}')

predictions = pd.DataFrame({'Player': data.loc[X_test.index, 'last_first'], 'True CPS': y_test, 'Predicted CPS': y_pred})
mispredictions = predictions[predictions['True CPS'] != predictions['Predicted CPS']]
print(mispredictions.head())
