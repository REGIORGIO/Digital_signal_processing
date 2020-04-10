from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import pandas as pd
#import tensorflow_utils as tf_utils

from skimage.feature import greycomatrix, greycoprops

import skimage.io
import os
import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
import keras
import keras.utils
from keras import utils as np_utils

# подготовка данных
Y = [1, 2, 3, 4, 5, 6, 7, 8]
# кодирование классов
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

print("dummy_y[0]: ", dummy_y[0])

''''
ill_to_index = {'Бурая ржавчина': 0,
                'Желтая ржавчина': 1,
                'Корневые гнили': 2,
                'Мучнистая роса': 3,
                'Пиренофороз': 4,
                'Полосатая мозайка': 5,
                'Септориоз': 6,
                'Снежная плесень': 7
                }

index_to_ill = {0: 'Бурая ржавчина',
                1: 'Желтая ржавчина',
                2: 'Корневые гнили',
                3: 'Мучнистая роса',
                4: 'Пиренофороз',
                5: 'Полосатая мозайка',
                6: 'Септориоз',
                7: 'Снежная плесень'
                }
'''

df_data = pd.DataFrame([[0., 0., 0., 0.]], [])              # saving parameters Haralick's
df_res = pd.DataFrame([[0., 0., 0., 0., 0., 0., 0., 0.]])   # saving labels

import skimage

image_index = 0
illness_index = 1
root = './test_dir/'

def generator_data(root):
    global image_index
    global illness_index
    global df_data
    global df_res
    for i, folder_dir in enumerate(os.listdir(root)):
        if folder_dir[0] == ".":
            continue

        print("Folder: ", folder_dir)
        img_RED_global = []
        img_GREEN_global = []
        img_BLUE_global = []
        for i, filename in enumerate(os.listdir(root + folder_dir)):
            path = root + folder_dir + "/" + filename
            if filename[0] == ".":
                continue
            print(" -Filename: ", filename)
            img_new = skimage.io.imread(path)
            # red
            img_red = img_new[:, :, 0]
            img_RED_global = img_red
            img_red = np.true_divide(img_red, 32)
            img_red = img_red.astype(int)
            glcm = greycomatrix(img_red, [1], [0], levels=8, symmetric=False, normed=True)

            df_data.loc[image_index] = [greycoprops(glcm, 'correlation')[0, 0],
                                        greycoprops(glcm, 'contrast')[0, 0],
                                        greycoprops(glcm, 'homogeneity')[0, 0],
                                        greycoprops(glcm, 'energy')[0, 0]]
            df_res.loc[image_index] = dummy_y[int(folder_dir) - 1]
            image_index += 1
            #  green
            img_green = img_new[:, :, 2]
            img_GREEN_global = img_green
            img_green = np.true_divide(img_green, 32)
            img_green = img_green.astype(int)
            glcm = greycomatrix(img_green, [1], [0], levels=8, symmetric=False, normed=True)

            df_data.loc[image_index] = [greycoprops(glcm, 'correlation')[0, 0],
                                        greycoprops(glcm, 'contrast')[0, 0],
                                        greycoprops(glcm, 'homogeneity')[0, 0],
                                        greycoprops(glcm, 'energy')[0, 0]]
            df_res.loc[image_index] = dummy_y[int(folder_dir) - 1]
            image_index += 1
            # blue
            img_blue = img_new[:, :, 0]
            img_BLUE_global = img_blue
            img_blue = np.true_divide(img_blue, 32)
            img_blue = img_blue.astype(int)
            glcm = greycomatrix(img_blue, [1], [0], levels=8, symmetric=False, normed=True)

            df_data.loc[image_index] = [greycoprops(glcm, 'correlation')[0, 0],
                                        greycoprops(glcm, 'contrast')[0, 0],
                                        greycoprops(glcm, 'homogeneity')[0, 0],
                                        greycoprops(glcm, 'energy')[0, 0]]
            df_res.loc[image_index] = dummy_y[int(folder_dir) - 1]
            image_index += 1
            # red-green
            img_r_g = img_RED_global - img_GREEN_global
            img_r_g = np.true_divide(img_r_g, 32)
            img_r_g = img_r_g.astype(int)
            glcm = greycomatrix(img_r_g, [1], [0], levels=8, symmetric=False, normed=True)

            df_data.loc[image_index] = [greycoprops(glcm, 'correlation')[0, 0],
                                        greycoprops(glcm, 'contrast')[0, 0],
                                        greycoprops(glcm, 'homogeneity')[0, 0],
                                        greycoprops(glcm, 'energy')[0, 0]]
            df_res.loc[image_index] = dummy_y[int(folder_dir) - 1]
            image_index += 1

            #  red-blue
            img_r_b = img_RED_global - img_BLUE_global
            img_r_b = np.true_divide(img_r_b, 32)
            img_r_b = img_r_b.astype(int)
            glcm = greycomatrix(img_r_b, [1], [0], levels=8, symmetric=False, normed=True)

            df_data.loc[image_index] = [greycoprops(glcm, 'correlation')[0, 0],
                                        greycoprops(glcm, 'contrast')[0, 0],
                                        greycoprops(glcm, 'homogeneity')[0, 0],
                                        greycoprops(glcm, 'energy')[0, 0]]
            df_res.loc[image_index] = dummy_y[int(folder_dir) - 1]
            image_index += 1
            # green-blue
            img_g_b = img_GREEN_global - img_BLUE_global
            img_g_b = np.true_divide(img_g_b, 32)
            img_g_b = img_g_b.astype(int)
            glcm = greycomatrix(img_g_b, [1], [0], levels=8, symmetric=False, normed=True)

            df_data.loc[image_index] = [greycoprops(glcm, 'correlation')[0, 0],
                                        greycoprops(glcm, 'contrast')[0, 0],
                                        greycoprops(glcm, 'homogeneity')[0, 0],
                                        greycoprops(glcm, 'energy')[0, 0]]
            df_res.loc[image_index] = dummy_y[int(folder_dir) - 1]
            image_index += 1

        illness_index += 1


generator_data('./test_dir/')
generator_data('./val_dir/')
generator_data('./train_dir/')

print(df_data.shape)
from sklearn.model_selection import train_test_split
df_data = df_data.to_numpy()
df_res = df_res.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(df_data, df_res, test_size=0.2, stratify=df_res, random_state=25)

model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, batch_size=2, verbose=1, validation_split=0.1)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Acc: ', test_acc)