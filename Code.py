CODING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("/content/dataset.csv")
data.columns


data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
data.head()


import datetime
import time
timestamp = []
for d, t in zip(data['Date'], data['Time']):
try:
ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
timestamp.append(time.mktime(ts.timetuple()))
except ValueError:
# print('ValueError')
timestamp.append('ValueError')
timeStamp = pd.Series(timestamp)
data['Timestamp'] = timeStamp.values
final_data = data.drop(['Date', 'Time'], axis=1)
final_data = final_data[final_data.Timestamp != 'ValueError']
final_data.head()


!pip install pyproj==1.9.6


!apt-get install libgeos-3.5.0
!apt-get install libgeos-dev
!pip install https://github.com/matplotlib/basemap/archive/master.zip


!pip install basemap
!pip install basemap-data

from mpl_toolkits.basemap import Basemap

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

longitudes = data["Longitude"].tolist()
latitudes = data["Latitude"].tolist()
#m = Basemap(width=12000000,height=9000000,projection='lcc',
#resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
x,y = m(longitudes,latitudes)

fig = plt.figure(figsize=(12,10))
plt.title("All affected areas")
m.plot(x, y, "o", markersize = 2, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()


X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, X_test.shape)


!pip install scikeras


from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Convert one-hot encoded labels to single binary labels
if y_train.ndim == 2 and y_train.shape[1] > 1:
y_train = np.argmax(y_train, axis=1)  # Convert to labels [0, 1]
if y_test.ndim == 2 and y_test.shape[1] > 1:
y_test = np.argmax(y_test, axis=1)


# Model creation function for binary classification
def create_model(neurons=16, activation='relu', optimizer='adam', loss='binary_crossentropy'):
model = Sequential()
model.add(Dense(neurons, activation=activation, input_shape=(3,)))  # Input shape for 3 features
model.add(Dense(neurons, activation=activation))
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron for binary classification
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
return model

# Wrapping the model creation function with KerasClassifier
model = KerasClassifier(model=create_model, verbose=1)

# Parameter grid for GridSearchCV
param_grid = {
'model__neurons': [8, 16],
'model__activation': ['relu'],
'epochs': [10],
'batch_size': [10]
}


# Create the GridSearchCV object
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1)

# Fit the grid search object with the data
grid_result = grid.fit(X_train, y_train)

# Print the best score and parameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))






# Evaluate the best model on test data
best_model = grid_result.best_estimator_.model_
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print("\nEvaluation result on Test Data: Loss = {}, accuracy = {}".format(test_loss, test_acc))
