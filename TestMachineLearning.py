import pandas as pd

white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data 
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')


# First rows of `red` 
#red.head()

# Last rows of `white`
#white.tail()

# Take a sample of 5 rows of `red`
#red.sample(5)

# Describe `white`
#white.describe()

# Double check for null values in `red`
#pd.isnull(red)

# Print info on white wine
#print(white.info())

# Print info on red wine
#print(red.info())

#import numpy as np
#print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
#print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))

#import matplotlib.pyplot as plt

#fig, ax = plt.subplots(1, 2)

#ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
#ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

#fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
#ax[0].set_ylim([0, 1000])
#ax[0].set_xlabel("Alcohol in % Vol")
#ax[0].set_ylabel("Frequency")
#ax[1].set_xlabel("Alcohol in % Vol")
#ax[1].set_ylabel("Frequency")
##ax[0].legend(loc='best')
##ax[1].legend(loc='best')
#fig.suptitle("Distribution of Alcohol in % Vol")

#plt.show()
# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)
#import seaborn as sns
#corr = wines.corr()
#sns.heatmap(corr, 
#            xticklabels=corr.columns.values,
#            yticklabels=corr.columns.values)
#sns.plt.show()

from sklearn.model_selection import train_test_split
import numpy as np

## Specify the data 
X=wines.ix[:,0:11]

## Specify the target labels and flatten the array 
y=np.ravel(wines.type)

## Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))
# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test,verbose=1)

print(score)