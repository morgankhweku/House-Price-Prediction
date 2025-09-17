import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load Data
df = pd.read_csv('kc_house_data.csv')
print (df.head())

#Data Visualization
print(df.isnull().sum())
print(df.describe().transpose())

plt.figure(figsize=(10,6))
sns.displot(df['price'])

sns.countplot(df['bedrooms'])

print(df.corr(numeric_only=True))
print(df.corr(numeric_only=True)['price'].sort_values())

plt.figure(figsize=(10,5))
sns.scatterplot(x='price',y= 'sqft_living', data=df)

plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y= 'long', data=df)
plt.show()

#Data Cleaning

df['date']= pd.to_datetime(df['date'])

#Function for year extraction from date
def year_extraction(date):
    return date.year

df['year'] = df['date'].apply(year_extraction)
df['month']= df['date'].apply(lambda  date: date.month)

print(df.groupby('month').mean()['price'])
print(df.groupby('year').mean()['price'])

df.drop(['id', 'date', 'zipcode'], axis=1, inplace=True)


print(df['yr_renovated'].value_counts())


#Data Preprocessing
X =df.drop('price',axis= 1).values
y = df['price'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Data Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Creating the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


model = Sequential()
model.add(Dense(19, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1))


model.compile(optimizer = 'adam', loss = 'mse')
model.summary()

model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=128,
    epochs=1000,
    callbacks=[early_stop]
)

losses = pd.DataFrame(model.history.history)
print(losses)
losses.plot()
plt.show()

#Data Evaluation
from sklearn.metrics import mean_absolute_error,mean_squared_error, explained_variance_score

predictions =model.predict(x_test)
print(predictions)

mse = mean_squared_error(y_test,predictions)
mae = mean_absolute_error(y_test,predictions)
e_var = explained_variance_score(y_test,predictions)

print("Mean Squared Error:",mse)
print("Mean Absolute Error:", mae)
print("Explained Variation:",e_var)

#Prediction Variation
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ideal line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Prediction vs Actual")
plt.tight_layout()
plt.show()


#Model to Predict Price of a new house
single_house = df.drop('price',axis= 1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(1, -1))
np.set_printoptions(suppress=True)

print(model.predict(single_house))