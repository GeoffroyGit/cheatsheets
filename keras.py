import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf # we use tensors rather than numpy arrays (for example tf.ones((3,3)) )

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Masking
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization, CategoryEncoding, Rescaling
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# to load images from a directory:
# tf.keras.preprocessing.image_dataset_from_directory


df = pd.DataFrame()

X = df[["feature one", "feature two"]]
X = df.drop(columns=["useless feature one", "useless feature two"])
y = df["target"]


# target encoding
y = to_categorical(y, num_classes=10)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# padding (for RNN)
X_pad = pad_sequences(X, dtype='float32', padding='post', value=0)

# create model
model = Sequential()

# preprocessing with tensorflow
normalizer = Normalization() # normalize
normalizer = Rescaling(scale=1./255.) # rescale
normalizer = CategoryEncoding() # categoricals
normalizer.adapt(X_train) # fit
model.add(normalizer) # add to the model (as if it was a pipe)

# add masking as the input layer (for RNN)
# input_dim must be the number of features in X
model.add(Masking(mask_value=0, input_shape=(4,3)))

# add imput layer
# input_dim must be the number of features in X
model.add(Dense(10, activation="relu", input_dim=4)) # dense input layer (here, for 4 features)
model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(225, 225, 3))) # convolutional input layer
model.add(MaxPooling2D(pool_size=(2,2))) # max pool after each convolutional layer

# add hidden layers
model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')) # convolutional layer
model.add(MaxPooling2D(pool_size=(2,2))) # max pool after each convolutional layer
model.add(AveragePooling2D(pool_size=(2, 2))) # average pool is an alternative to max pool
model.add(Flatten()) # needed between convolutional layer and dense layer
model.add(Dense(10, activation="relu")) # dense layer
model.add(Dropout(rate=0.2)) # dropout

model.add(LSTM(units=10, return_sequences=True, activation='tanh')) # LSTM recurent layer
model.add(GRU(units=10, return_sequences=True, activation='tanh')) # GRU recurent layer
model.add(SimpleRNN(3, return_sequences=True)) # simple recurent layer
model.add(SimpleRNN(3, return_sequences=False)) # simple recurent layer
model.add(Dense(10, activation="relu")) # dense layer
model.add(Dropout(rate=0.2)) # dropout

# add output layer
model.add(Dense(1, activation="linear")) # for regression (here, predict one target)
model.add(Dense(10, activation='linear')) # for regression (here, predict 10 targets)
model.add(Dense(1, activation="sigmoid")) # for binary classification
model.add(Dense(5, activation="softmax")) # for classification (here, 5 categories)

# summary
model.summary()

# compile model
model.compile(optimizer="adam", loss="mse", metrics=["mae"]) # for regression (with metrics: MSE, MAE, RMSE, RMSLE, R-squared, etc)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) # for binary classification
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) # for classification (with metrics: Precision, Recall, Accuracy, F1 score, etc)
model.compile(optimizer='rmsprop', loss='mse', metrics=["mae"])  # optimizer for RNN

# Custom metrics or losses
def custom_mse(y_true, y_pred):
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff)
model.compile(metrics=[custom_mse]) # custom metric
model.compile(loss=custom_mse) # custom loss

# custom optimizer
custom_opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99)
model.compile(optimizer=custom_opt)

# fit model
es = EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=16, epochs=100, validation_split=0.3, callbacks=[es], shuffle=True)
#plot_history(history)

# evaluate model
model.evaluate(X_test, y_test)

# make prediction
y_pred = model.predict(X_test)

# save
models.save_model(model, '/path/to/model')
# load
model = models.load_model('/path/to/model')


# regularizers (apply a penalty)

reg_l1 = regularizers.L1(0.01)
reg_l2 = regularizers.L2(0.01)
reg_l1_l2 = regularizers.l1_l2(l1=0.005, l2=0.0005)

model = Sequential()

model.add(layers.Dense(100, activation='relu', input_dim=13))
model.add(layers.Dense(50, activation='relu', kernel_regularizer=reg_l1))
model.add(layers.Dense(20, activation='relu', bias_regularizer=reg_l2))
model.add(layers.Dense(10, activation='relu', activity_regularizer=reg_l1_l2))
model.add(layers.Dense(1, activation='sigmoid'))


# ------------------------
# complete example 1 (CNN)
# ------------------------

model = Sequential()

model.add(Conv2D(32, (5, 5),
                 padding='same',
                 strides = (1,1),
                 input_shape=(32, 32, 3),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))

# ------------------------
# complete example 2 (CNN)
# ------------------------

from tensorflow.keras.datasets import mnist

# load mnist data
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

model_pipe = Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28,28)),
    layers.experimental.preprocessing.Rescaling(scale=1./255.),
    layers.Conv2D(16, (3,3), padding='same', activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(32, (2,2), padding='same', activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(50, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model_pipe.compile(loss='sparse_categorical_crossentropy', # No need to OHE target
              optimizer='adam',
              metrics=['accuracy'])

model_pipe.fit(X_train_raw, y_train_raw,
          epochs=1,  # Use early stopping in practice
          batch_size=32,
          verbose=1)

# We can now evaluate the model on the test data
print(model_pipe.evaluate(X_test_raw, y_test_raw, verbose=0))

# show output from a specific layer (here, last kernel of first layer)
layer_1 = model.layers[0]
plt.imshow(layer_1.weights[0][:,:,:,15], cmap='gray');

# ---------------------------------
# example : transfer learning (CNN)
# ---------------------------------

from tensorflow.keras.applications import vgg16

model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

model.trainable = False

flatten_layer = Flatten()
dense_layer = Dense(100, activation='relu')
prediction_layer = Dense(10, activation='softmax')

model = Sequential([
    model,
    flatten_layer,
    dense_layer,
    prediction_layer
])
