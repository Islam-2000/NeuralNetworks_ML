'''Imports'''
from keras.datasets.cifar10 import load_data
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

'''Load Dataset'''
(X_train, y_train), (X_test, y_test) = load_data()

'''Sample Images'''
# for i in range(49):
#     plt.subplot(7, 7, 1 + i)
#     plt.axis('off')
#     plt.imshow(X_train[i])
# plt.show()

'''Model Creation'''
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(LeakyReLU(alpha=0.2))

model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
model.add(LeakyReLU(alpha=0.2))

model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
model.add(LeakyReLU(alpha=0.2))

model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
model.add(LeakyReLU(alpha=0.2))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(lr=0.0002, beta_1=0.5)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

