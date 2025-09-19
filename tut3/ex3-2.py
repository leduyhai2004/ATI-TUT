import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Load csv file into the dataframe: df
df = pd.read_csv("hourly_wages.csv")

# Split the dataframe df into two dataframes:
wagePerHourDf = df.iloc[:, 0]
predictorsDf = df.iloc[:, 1:df.shape[1]]

# Create predictors NumPy array: predictors
predictors = predictorsDf.to_numpy()

# Create target NumPy array: target
target = wagePerHourDf.to_numpy()

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model
model = Sequential()

# Add the first hidden layer (e.g., 50 neurons, relu activation)
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second hidden layer (e.g., 32 neurons, relu activation)
model.add(Dense(32, activation='relu'))

# Add the output layer (1 neuron since predicting continuous value)
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target, epochs=20, batch_size=32, verbose=1)
