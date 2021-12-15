'''Imports'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import BatchNormalization, Flatten, Convolution2D, Convolution2DTranspose, Dropout, \
    MaxPool2D, Dense
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils.np_utils import to_categorical

'''Hyperparameters'''
width, height = 32, 32
batch_size = 1024
epochs = 20
num_classes = 10
validation_split = 0.25
verbosity = 1

'''Load Dataset'''
(X_all_train, y_all_train), (X_test, y_test) = cifar10.load_data()

'''Parse to float32 & Normalize'''
X_all_train = X_all_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

no_classes = len(np.unique(y_all_train))

'''Encoding'''
y_all_train = to_categorical(y_all_train, no_classes)
y_test = to_categorical(y_test, no_classes)

'''Model Creation & Summary'''
model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='he_normal', input_shape=(32, 32, 3)))
model.add(Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(16, kernel_size=3, activation='relu', kernel_initializer='he_normal'))

model.add(Conv2DTranspose(16, kernel_size=3, activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(32, kernel_size=3, activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(64, kernel_size=3, activation='relu', kernel_initializer='he_normal'))

model.add(Conv2D(3, kernel_size=3, activation='softmax', padding='same'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

'''Fitting'''
history = model.fit(X_all_train, X_all_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split)

'''Plot Accuracy & Loss'''
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss'], loc='upper left')
plt.show()

pd.DataFrame(history.history).plot()

'''Prediction'''
print("Starting Testing...")
model.evaluate(X_test, X_test)

pred = model.predict(X_test)

'''Weights'''
for w in model.trainable_weights:
    print(K.eval(w))

'''Plotting Sample Reconstructions'''
num_reconstructions = 8
samples = X_test[:num_reconstructions]
targets = y_test[:num_reconstructions]
reconstructions = model.predict(samples)

for i in np.arange(0, num_reconstructions):
    sample = samples[i][:, :, 0]
    reconstruction = reconstructions[i][:, :, 0]
    input_class = targets[i]
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(sample)
    axes[0].set_title('Original image')
    axes[1].imshow(reconstruction)
    axes[1].set_title('Reconstruction with Conv2DTranspose')
    fig.suptitle(f'CIFAR Target = {input_class}')
    plt.show()
