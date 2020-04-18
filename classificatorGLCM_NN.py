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


def get_nth_sum(df_data, i, n, key, SKO):
    if n == 2:
        return SKO * (df_data[i-1][key] + df_data[i][key]) / n
    elif n == 3:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key]) / n
    elif n == 4:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key] + df_data[i+2][key]) / n


def generate_haralick_params(df_data):
    df_data_new = []
    for i in range(5):
        for i in range(1, len(df_data) - 2):
            if df_data[i]['class'] == df_data[i-1]['class'] == df_data[i+1]['class'] == df_data[i+2]['class']:

                SKO = random.randint(95, 105) / 100

                red_correlation = get_nth_sum(df_data, i, 4, 'red_correlation', SKO)
                red_contrast = get_nth_sum(df_data, i, 4, 'red_contrast', SKO)
                red_homogeneity = get_nth_sum(df_data, i, 4, 'red_homogeneity', SKO)
                red_energy = get_nth_sum(df_data, i, 4, 'red_energy', SKO)

                green_correlation = get_nth_sum(df_data, i, 4, 'green_correlation', SKO)
                green_contrast = get_nth_sum(df_data, i, 4, 'green_contrast', SKO)
                green_homogeneity = get_nth_sum(df_data, i, 4, 'green_homogeneity', SKO)
                green_energy = get_nth_sum(df_data, i, 4, 'green_energy', SKO)

                blue_correlation = get_nth_sum(df_data, i, 4, 'blue_correlation', SKO)
                blue_contrast = get_nth_sum(df_data, i, 4, 'blue_contrast', SKO)
                blue_homogeneity = get_nth_sum(df_data, i, 4, 'blue_homogeneity', SKO)
                blue_energy = get_nth_sum(df_data, i, 4, 'blue_energy', SKO)

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

        for i in range(1, len(df_data) - 1):
            if df_data[i]['class'] == df_data[i-1]['class'] == df_data[i+1]['class']:

                SKO = random.randint(95, 105) / 100

                red_correlation = get_nth_sum(df_data, i, 3, 'red_correlation', SKO)
                red_contrast = get_nth_sum(df_data, i, 3, 'red_contrast', SKO)
                red_homogeneity = get_nth_sum(df_data, i, 3, 'red_homogeneity', SKO)
                red_energy = get_nth_sum(df_data, i, 3, 'red_energy', SKO)

                green_correlation = get_nth_sum(df_data, i, 3, 'green_correlation', SKO)
                green_contrast = get_nth_sum(df_data, i, 3, 'green_contrast', SKO)
                green_homogeneity = get_nth_sum(df_data, i, 3, 'green_homogeneity', SKO)
                green_energy = get_nth_sum(df_data, i, 3, 'green_energy', SKO)

                blue_correlation = get_nth_sum(df_data, i, 3, 'blue_correlation', SKO)
                blue_contrast = get_nth_sum(df_data, i, 3, 'blue_contrast', SKO)
                blue_homogeneity = get_nth_sum(df_data, i, 3, 'blue_homogeneity', SKO)
                blue_energy = get_nth_sum(df_data, i, 3, 'blue_energy', SKO)

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
            if df_data[i]['class'] == df_data[i - 1]['class']:

                SKO = random.randint(95, 105) / 100

                red_correlation = get_nth_sum(df_data, i, 2, 'red_correlation', SKO)
                red_contrast = get_nth_sum(df_data, i, 2, 'red_contrast', SKO)
                red_homogeneity = get_nth_sum(df_data, i, 2, 'red_homogeneity', SKO)
                red_energy = get_nth_sum(df_data, i, 2, 'red_energy', SKO)

                green_correlation = get_nth_sum(df_data, i, 2, 'green_correlation', SKO)
                green_contrast = get_nth_sum(df_data, i, 2, 'green_contrast', SKO)
                green_homogeneity = get_nth_sum(df_data, i, 2, 'green_homogeneity', SKO)
                green_energy = get_nth_sum(df_data, i, 2, 'green_energy', SKO)

                blue_correlation = get_nth_sum(df_data, i, 2, 'blue_correlation', SKO)
                blue_contrast = get_nth_sum(df_data, i, 2, 'blue_contrast', SKO)
                blue_homogeneity = get_nth_sum(df_data, i, 2, 'blue_homogeneity', SKO)
                blue_energy = get_nth_sum(df_data, i, 2, 'blue_energy', SKO)

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

    return df_data_new + df_data


def get_optimal_params(x_train, y_train, x_test, y_test):
    parameter_space = {
        'alpha': [0.0001, 0.001, 0.01],
        'hidden_layer_sizes': [(30, 30), (40, 40), (50, 50), (100, 100), (100,)],
        'activation': ['relu', 'logistic'],
        'solver': ['adam', 'lbfgs'],
        'max_iter': [600, 700, 800, 900]
    }
    mlp = MLPClassifier(verbose=0)

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
    clf.fit(x_train, y_train)
    print('Best parameters found:\n', clf.best_params_)

    means = clf.cv_results_['mean_test_score']
    print(max(means))
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    accuracy = clf.score(x_test, y_test)
    print("Accuracy = {}\n".format(accuracy))
    target_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)

    # return clf.best_params_


def get_optimal(x_train, y_train, x_test, y_test):
    parameter_space = {
        'alphas': [0.0001, 0.001, 0.01],
        'hidden_layer_sizes': [(30, 30), (40, 40), (50, 50), (100, 100), (100,)],
        'activations': ['relu', 'logistic'],
        'solvers': ['adam', 'lbfgs'],
        'max_iters': [800, 900, 1000, 1200]
    }
    best_accuracy = 0
    best_params = {'activation': 0,
                   'alpha': 0,
                   'hidden_layer_sizes': (1,),
                   'solver': '',
                   'max_iter': 0
                   }

    for alpha in parameter_space['alphas']:
        for hidden_layer_size in parameter_space['hidden_layer_sizes']:
            for activation in  parameter_space['activations']:
                for solver in parameter_space['solvers']:
                    for max_iter in parameter_space['max_iters']:
                        clf = MLPClassifier(alpha=alpha,
                                            hidden_layer_sizes=hidden_layer_size,
                                            activation=activation,
                                            solver=solver,
                                            max_iter=max_iter
                                            )
                        clf.fit(x_train, y_train)

                        accuracy = clf.score(x_test, y_test)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params['activation'] = activation
                            best_params['alpha'] = alpha
                            best_params['hidden_layer_sizes'] = hidden_layer_size
                            best_params['solver'] = solver
                            best_params['max_iter'] = max_iter


                        # print('{} <-- Activation = {} '
                        #       'Alpha = {} Hidden_layer_sizes = {}'
                        #       ' Solver = {} Epoch = {}'.format(accuracy, activation, alpha, hidden_layer_size, solver, max_iter))
    print('\nBest params:')
    print('{} <-- Activation = {} '
          'Alpha = {} Hidden_layer_sizes = {}'
          ' Solver = {} Epoch = {}'.format(best_accuracy, best_params['activation'],
                                           best_params['alpha'], best_params['hidden_layer_sizes'],
                                           best_params['solver'], best_params['max_iter']))


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
    # get_optimal(x_train, y_train, x_test, y_test)
    # get_optimal_params(x_train, y_train, x_test, y_test)

    # print('Оптимальные параметры найдены: {}'.format(optimal_params))
    print('Train size = {}, test size = {}'.format(len(y_train), len(y_test)))
    # # clf = MLPClassifier(activation=optimal_params['activation'],
    # #                     alpha=optimal_params['alpha'],
    # #                     hidden_layer_sizes=optimal_params['hidden_layer_sizes'],
    # #                     max_iter=optimal_params['max_iter'],
    # #                     solver=optimal_params['solver'])
    clf = MLPClassifier(activation='logistic',
                        max_iter=15000,
                        solver='lbfgs')


    clf.fit(x_train, y_train)

    accuracy = clf.score(x_test, y_test)
    print("Accuracy = {}\n".format(accuracy))
    target_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)