from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tensorflow.keras.models import Model
from keras.layers import *
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

import collections
import imblearn


def createModel(inputA):
    # Block - 1
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputA)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    # Block - 2

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    # Block - 3

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    # Block - 4

    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    # Block - 5

    x = Flatten()(x)
    x = Dense(64, kernel_initializer='he_normal')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Block - 6

    x = Dense(64, kernel_initializer='he_normal')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return x


def trainModel():
    data = pd.read_csv('resources/fer2013.csv')

    data.Usage.unique()

    train_data = data[data.Usage == 'Training']
    val_data = data[data.Usage == 'PublicTest']
    test_data = data[data.Usage == 'PrivateTest']

    oversampler = imblearn.over_sampling.RandomOverSampler()

    print(collections.Counter(train_data.emotion))

    x_train, y_train = oversampler.fit_resample(train_data.pixels.values.reshape(-1, 1), train_data.emotion.values)

    x_val = val_data.pixels.values.reshape(-1, 1)
    y_val = val_data.emotion.values

    x_test = test_data.pixels.values.reshape(-1, 1)
    y_test = test_data.emotion.values

    collections.Counter(y_train)

    x_train = list(x_train)
    x_val = list(x_val)
    x_test = list(x_test)

    for i, item in enumerate(x_train):
        x_train[i] = np.fromstring(item[0], sep=' ').reshape(48, 48, 1)
    for i, item in enumerate(x_val):
        x_val[i] = np.fromstring(item[0], sep=' ').reshape(48, 48, 1)
    for i, item in enumerate(x_test):
        x_test[i] = np.fromstring(item[0], sep=' ').reshape(48, 48, 1)

    x_train = np.vstack(x_train).reshape(-1, 48, 48, 1)
    x_val = np.vstack(x_val).reshape(-1, 48, 48, 1)
    x_test = np.vstack(x_test).reshape(-1, 48, 48, 1)

    y_train = to_categorical(y_train, num_classes=7)
    y_val = to_categorical(y_val, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    num_classes = 7
    batch_size = 64

    inputA = Input(shape=(48, 48, 1))
    x = createModel(inputA)

    # Block - 7
    output = Dense(num_classes, kernel_initializer='he_normal')(x)
    output = Activation('softmax')(output)

    model = Model(inputs=inputA, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

    nb_train_samples = 50505
    nb_validation_samples = 3589
    epochs = 25

    checkpoint = ModelCheckpoint('EmotionDetectionModel2.h5',
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True,
                                 verbose=1)
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1,
                              restore_best_weights=True
                              )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=3,
                                  verbose=1,
                                  min_delta=0.0001)

    callbacks = [earlystop, checkpoint, reduce_lr]

    history = model.fit(
        x_train, y_train,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, y_val),
        validation_steps=nb_validation_samples // batch_size,
        batch_size=32)
    model.evaluate(x_test, y_test, batch_size=32)