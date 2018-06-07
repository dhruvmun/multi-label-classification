import keras
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation
from keras import backend as K

class Model:
    @staticmethod
    def model(length, width, channels, classes):
        input_shape = (length, width, channels)
        
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same', input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(32, (3,3), padding='same', input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3,3), padding='same', input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(32, (3,3), padding='same', input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3,3), padding='same', input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='sigmoid'))
        
        return model