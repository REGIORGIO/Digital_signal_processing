import pandas as pd
from skimage.feature import greycomatrix, greycoprops
import skimage
import skimage.io
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import random


def generator_data(root):
    image_index = 0
    illness_index = 1

    df_data = []  # saving parameters Haralick's

    for i, folder_dir in enumerate(os.listdir(root)):
        if folder_dir[0] == ".":
            continue

        print("Folder: ", folder_dir)
        for _, filename in enumerate(os.listdir(root + folder_dir)):
            path = root + folder_dir + "/" + filename
            if filename[0] == ".":
                continue
            img_new = skimage.io.imread(path) #Загружаем очередное изображение по указанному пути
            print(" -Filename: {}, Original Image Shape:{} ".format(filename, img_new.shape)) #выводим разрешение изображения и цвета(?)
            # print(img_new)

            # red
            img_red = img_new[:, :, 0]
            # print(img_red)
            # print(img_red.shape)
            # print(print(" -Filename: {}, RED Image Shape:{} ".format(filename, img_red.shape)))
            img_RED_global = img_red
            img_red = np.true_divide(img_red, 32)
            img_red = img_red.astype(int)
            glcm = greycomatrix(img_red, [1], [0], levels=8, symmetric=False, normed=True)

            red_correlation = greycoprops(glcm, 'correlation')[0, 0]
            red_contrast =  greycoprops(glcm, 'contrast')[0, 0]
            red_homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
            red_energy = greycoprops(glcm, 'energy')[0, 0]

            image_index += 1
            #
            #  green
            img_green = img_new[:, :, 1]
            # print(print(" -Filename: {}, GREEN Image Shape:{} ".format(filename, img_green.shape)))
            img_GREEN_global = img_green
            img_green = np.true_divide(img_green, 32)
            img_green = img_green.astype(int)
            glcm = greycomatrix(img_green, [1], [0], levels=8, symmetric=False, normed=True)

            green_correlation = greycoprops(glcm, 'correlation')[0, 0]
            green_contrast = greycoprops(glcm, 'contrast')[0, 0]
            green_homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
            green_energy = greycoprops(glcm, 'energy')[0, 0]
            image_index += 1
            #
            # blue
            img_blue = img_new[:, :, 2]
            # print(print(" -Filename: {}, BLUE Image Shape:{} ".format(filename, img_blue.shape)))
            img_BLUE_global = img_blue
            img_blue = np.true_divide(img_blue, 32)
            img_blue = img_blue.astype(int)
            glcm = greycomatrix(img_blue, [1], [0], levels=8, symmetric=False, normed=True)

            blue_correlation = greycoprops(glcm, 'correlation')[0, 0]
            blue_contrast = greycoprops(glcm, 'contrast')[0, 0]
            blue_homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
            blue_energy = greycoprops(glcm, 'energy')[0, 0]

            df_data.append({'red_correlation': red_correlation,
                            'red_contrast': red_contrast,
                            'red_homogeneity': red_homogeneity,
                            'red_energy': red_energy,
                            'green_correlation': green_correlation,
                            'green_contrast': green_contrast,
                            'green_homogeneity': green_homogeneity,
                            'green_energy': green_energy,
                            'blue_correlation': blue_correlation,
                            'blue_contrast': blue_contrast,
                            'blue_homogeneity': blue_homogeneity,
                            'blue_energy': blue_energy,
                            'class': int(folder_dir)})

            image_index += 1
        illness_index += 1
    return df_data


def generate_haralick_params(df_data):
    df_data_new = []
    for i in range(1, len(df_data) - 1):
        if df_data[i]['class'] == df_data[i-1]['class'] == df_data[i+1]['class']:

            CKO = random.randint(95, 105) / 100

            red_correlation = CKO * (df_data[i-1]['red_correlation'] + df_data[i]['red_correlation'] + df_data[i+1]['red_correlation'])/3
            red_contrast = CKO * (df_data[i - 1]['red_contrast'] + df_data[i]['red_contrast'] + df_data[i + 1]['red_contrast']) / 3
            red_homogeneity = CKO * (df_data[i - 1]['red_homogeneity'] + df_data[i]['red_homogeneity'] + df_data[i + 1]['red_homogeneity']) / 3
            red_energy = CKO * (df_data[i - 1]['red_energy'] + df_data[i]['red_energy'] + df_data[i + 1]['red_energy']) / 3

            green_correlation = CKO * (df_data[i - 1]['green_correlation'] + df_data[i]['green_correlation'] + df_data[i + 1]['green_correlation']) / 3
            green_contrast = CKO * (df_data[i - 1]['green_contrast'] + df_data[i]['green_contrast'] + df_data[i + 1]['green_contrast']) / 3
            green_homogeneity = CKO * (df_data[i - 1]['green_homogeneity'] + df_data[i]['green_homogeneity'] + df_data[i + 1]['green_homogeneity']) / 3
            green_energy = CKO * (df_data[i - 1]['green_energy'] + df_data[i]['green_energy'] + df_data[i + 1]['green_energy']) / 3

            blue_correlation = CKO * (df_data[i - 1]['blue_correlation'] + df_data[i]['blue_correlation'] + df_data[i + 1]['blue_correlation']) / 3
            blue_contrast = CKO * (df_data[i - 1]['blue_contrast'] + df_data[i]['blue_contrast'] + df_data[i + 1]['blue_contrast']) / 3
            blue_homogeneity = CKO * (df_data[i - 1]['blue_homogeneity'] + df_data[i]['blue_homogeneity'] + df_data[i + 1]['blue_homogeneity']) / 3
            blue_energy = CKO * (df_data[i - 1]['blue_energy'] + df_data[i]['blue_energy'] + df_data[i + 1]['blue_energy']) / 3

            cl = df_data[i]['class']

            df_data_new.append({'red_correlation': red_correlation,
                                'red_contrast': red_contrast,
                                'red_homogeneity': red_homogeneity,
                                'red_energy': red_energy,
                                'green_correlation': green_correlation,
                                'green_contrast': green_contrast,
                                'green_homogeneity': green_homogeneity,
                                'green_energy': green_energy,
                                'blue_correlation': blue_correlation,
                                'blue_contrast': blue_contrast,
                                'blue_homogeneity': blue_homogeneity,
                                'blue_energy': blue_energy,
                                'class': cl
                                })
        else:
            continue

    for i in range(1, len(df_data)):
        if df_data[i]['class'] == df_data[i-1]['class']:

            CKO = random.randint(95, 105) / 100

            red_correlation = CKO * (df_data[i-1]['red_correlation'] + df_data[i]['red_correlation']) / 2
            red_contrast = CKO * (df_data[i - 1]['red_contrast'] + df_data[i]['red_contrast']) / 2
            red_homogeneity = CKO * (df_data[i - 1]['red_homogeneity'] + df_data[i]['red_homogeneity']) / 2
            red_energy = CKO * (df_data[i - 1]['red_energy'] + df_data[i]['red_energy']) / 2

            green_correlation = CKO * (df_data[i - 1]['green_correlation'] + df_data[i]['green_correlation']) / 2
            green_contrast = CKO * (df_data[i - 1]['green_contrast'] + df_data[i]['green_contrast']) / 2
            green_homogeneity = CKO * (df_data[i - 1]['green_homogeneity'] + df_data[i]['green_homogeneity']) / 2
            green_energy = CKO * (df_data[i - 1]['green_energy'] + df_data[i]['green_energy']) / 2

            blue_correlation = CKO * (df_data[i - 1]['blue_correlation'] + df_data[i]['blue_correlation']) / 2
            blue_contrast = CKO * (df_data[i - 1]['blue_contrast'] + df_data[i]['blue_contrast']) / 2
            blue_homogeneity = CKO * (df_data[i - 1]['blue_homogeneity'] + df_data[i]['blue_homogeneity']) / 2
            blue_energy = CKO * (df_data[i - 1]['blue_energy'] + df_data[i]['blue_energy']) / 2

            cl = df_data[i]['class']

            df_data_new.append({'red_correlation': red_correlation,
                                'red_contrast': red_contrast,
                                'red_homogeneity': red_homogeneity,
                                'red_energy': red_energy,
                                'green_correlation': green_correlation,
                                'green_contrast': green_contrast,
                                'green_homogeneity': green_homogeneity,
                                'green_energy': green_energy,
                                'blue_correlation': blue_correlation,
                                'blue_contrast': blue_contrast,
                                'blue_homogeneity': blue_homogeneity,
                                'blue_energy': blue_energy,
                                'class': cl
                                })
        else:
            continue

    return df_data_new


if __name__ == "__main__":
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

    df_test = pd.DataFrame(generator_data('./test_dir/'))
    df_val = pd.DataFrame(generator_data('./val_dir/'))
    df_train = (generator_data('./train_dir/'))

    #
    # увеличиваем тестовый датасет
    df_test = pd.concat([df_test, df_val])

    x_test = df_test.iloc[:, 0:12]
    y_test = df_test.iloc[:, 12]
    # # #

    #
    #
    # df_train = pd.DataFrame(generator_data('./train_dir/'))
    # x_train = df_train.iloc[:, 0:12]
    # y_train = df_train.iloc[:, 12]
    #

    df_train = pd.DataFrame(generate_haralick_params(df_train))

    x_train = df_train.iloc[:, 0:12]
    y_train = df_train.iloc[:, 12]
    #




    parameter_space = {
        'alpha': [0.0001, 0.001, 0.01],
        'hidden_layer_sizes': [(30, 30), (40, 40), (50, 50), (100, 100), (100,)],
        'activation': ['relu', 'logistic'],
        'solver': ['lbfgs'],
        'max_iter': [600, 700, 800, 900, 1000]
    }
    mlp = MLPClassifier(verbose=0)

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(x_train, y_train)
    print('Best parameters found:\n', clf.best_params_)

    means = clf.cv_results_['mean_test_score']
    print(max(means))
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    #
    clf = MLPClassifier(activation='logistic',
                        alpha=0.01,
                        hidden_layer_sizes=(30, 30),
                        max_iter=1000,
                        solver='lbfgs')
    clf.fit(x_train, y_train)

    print('Train size = {}, test size = {}'.format(len(y_train), len(y_test)))

    accuracy = clf.score(x_test, y_test)
    print("Accuracy = {}\n".format(accuracy))
    target_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)