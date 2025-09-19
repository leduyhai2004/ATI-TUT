import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load csv file into the dataframe: df
df = pd.read_csv("titanic_all_numeric.csv")

# Convert the boolean values of the 'age_was_missing' column to integer
df.age_was_missing = df.age_was_missing.replace({True: 1, False: 0})

# Create predictors NumPy array: predictors
predictors = df.drop(['survived'], axis=1).values

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Convert the target to categorical: target
target = to_categorical(df['survived'])

# Define the input shape: input_shape
input_shape = (n_cols,)

# Define a function to create model_1:
def get_new_model1(input_shape):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=input_shape))
    model.add(Dense(2, activation='softmax'))
    return model

# Define a function to create model_2:
def get_new_model2(input_shape):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

# Specify the model_1
model_1 = get_new_model1(input_shape)

# Specify the model_2
model_2 = get_new_model2(input_shape)

# Compile the model_1
model_1.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Compile the model_2
model_2.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Define early_stopping_monitor with patience = 5
early_stopping_monitor = EarlyStopping(patience=5)

# Fit model_1
model_1_training = model_1.fit(predictors, target,
                               epochs=20,
                               validation_split=0.2,
                               callbacks=[early_stopping_monitor],
                               verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target,
                               epochs=20,
                               validation_split=0.2,
                               callbacks=[early_stopping_monitor],
                               verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', label="Model_1 (1 hidden layer)")
plt.plot(model_2_training.history['val_loss'], 'b', label="Model_2 (3 hidden layers)")
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()
