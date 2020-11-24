import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

data_dir = "./VideoFile/"
seq_len = 16

classes = ["punching", "kicking"]


#  Creating frames from videos

def frames_extraction(video_path):
    frames_list = []

    vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable 
    count = 1

    while count <= seq_len:

        success, image = vidObj.read()
        if success:
            if(count%16==0):
                print("ㅊㅋ")
            image = cv2.resize(image, (224, 224))
            frames_list.append(image)
            count += 1
        else:
            print("Defected frame")
            break

    vidObj.release()
    return frames_list

def create_data(input_dir):
    X = []
    Y = []

    classes_list = os.listdir(input_dir)

    for c in classes_list:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c))
        for f in files_list:
            frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
            if len(frames) == seq_len:
                X.append(frames)

                y = [0] * len(classes)
                y[classes.index(c)] = 1
                Y.append(y)


    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y
"""
X, Y = create_data("./VideoFile/")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)
"""
model = Sequential()

model.add(TimeDistributed(Conv2D(64, 5, padding='same', input_shape=(seq_len, 224, 224, 3))))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))

model.add(TimeDistributed(Conv2D(64, 5, padding='same')))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(32))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = keras.optimizers.SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
"""
earlystop = EarlyStopping(patience=7)
callbacks = [earlystop]

history = model.fit(x=X_train, y=y_train, epochs=40, batch_size=8, shuffle=True, validation_split=0.2,
                    callbacks=callbacks)
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred))"""