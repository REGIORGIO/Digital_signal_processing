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
import cv2 as cv
from statistics import mean
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def generator_data(root):
    image_index = 0
    illness_index = 1
    masks = [[132, 0, 0],
             [204, 208, 193],
             [245, 228, 103],
             [54, 46, 34],
             [182, 146, 82],
             [0, 255, 0]]
    r_coef, g_coef, b_coef = (0.3, 0.59, 0.11)

    offset = [20]
    angle = [-np.pi / 2]
    levels = 8

    df_data = []  # saving parameters Haralick's

    for i, folder_dir in enumerate(os.listdir(root)):
        if folder_dir[0] == ".":
            continue

        # print("Folder: ", folder_dir)
        for _, filename in enumerate(os.listdir(root + folder_dir)):
            path = root + folder_dir + "/" + filename
            if filename[0] == ".":
                continue

            image = cv.imread(path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = np.array(image, dtype=np.int64)
            masked_images = np.zeros((6,) + image.shape[:-1])
            for i, mask in enumerate(masks):
                masked_images[i] = (r_coef * (1 - (np.abs(image[:, :, 0] - mask[0])) / max(mask[0], 255 - mask[0]))
                                    + g_coef * (1 - (np.abs(image[:, :, 1] - mask[1])) / max(mask[1], 255 - mask[1]))
                                    + b_coef * (1 - (np.abs(image[:, :, 2] - mask[2])) / max(mask[2], 255 - mask[2])))
            masked_images = np.array(masked_images * (levels - 1), dtype=np.uint16)
            glcms = np.zeros((len(masks), levels, levels, 1, 1))

            for i in range(len(masks)):
                glcms[i] = greycomatrix(masked_images[i], offset, angle, levels=levels, symmetric=False, normed=True)

            correlation1 = greycoprops(glcms[0], 'correlation')[0, 0]
            contrast1 = greycoprops(glcms[0], 'contrast')[0, 0]
            homogeneity1 = greycoprops(glcms[0], 'homogeneity')[0, 0]
            energy1 = greycoprops(glcms[0], 'energy')[0, 0]
            asm1 = greycoprops(glcms[0], 'ASM')[0, 0]

            correlation2 = greycoprops(glcms[1], 'correlation')[0, 0]
            contrast2 = greycoprops(glcms[1], 'contrast')[0, 0]
            homogeneity2 = greycoprops(glcms[1], 'homogeneity')[0, 0]
            energy2 = greycoprops(glcms[1], 'energy')[0, 0]
            asm2 = greycoprops(glcms[1], 'ASM')[0, 0]

            correlation3 = greycoprops(glcms[2], 'correlation')[0, 0]
            contrast3 = greycoprops(glcms[2], 'contrast')[0, 0]
            homogeneity3 = greycoprops(glcms[2], 'homogeneity')[0, 0]
            energy3 = greycoprops(glcms[2], 'energy')[0, 0]
            asm3 = greycoprops(glcms[2], 'ASM')[0, 0]

            correlation4 = greycoprops(glcms[3], 'correlation')[0, 0]
            contrast4 = greycoprops(glcms[3], 'contrast')[0, 0]
            homogeneity4 = greycoprops(glcms[3], 'homogeneity')[0, 0]
            energy4 = greycoprops(glcms[3], 'energy')[0, 0]
            asm4 = greycoprops(glcms[3], 'ASM')[0, 0]

            correlation5 = greycoprops(glcms[4], 'correlation')[0, 0]
            contrast5 = greycoprops(glcms[4], 'contrast')[0, 0]
            homogeneity5 = greycoprops(glcms[4], 'homogeneity')[0, 0]
            energy5 = greycoprops(glcms[4], 'energy')[0, 0]
            asm5 = greycoprops(glcms[4], 'ASM')[0, 0]

            correlation6 = greycoprops(glcms[5], 'correlation')[0, 0]
            contrast6 = greycoprops(glcms[5], 'contrast')[0, 0]
            homogeneity6 = greycoprops(glcms[5], 'homogeneity')[0, 0]
            energy6 = greycoprops(glcms[5], 'energy')[0, 0]
            asm6 = greycoprops(glcms[5], 'ASM')[0, 0]




            df_data.append({'correlation1': correlation1,
                            'contrast1': contrast1,
                            'homogeneity1': homogeneity1,
                            'energy1': energy1,
                            'asm1': asm1,

                            'correlation2': correlation2,
                            'contrast2': contrast2,
                            'homogeneity2': homogeneity2,
                            'energy2': energy2,
                            'asm2': asm2,

                            'correlation3': correlation3,
                            'contrast3': contrast3,
                            'homogeneity3': homogeneity3,
                            'energy3': energy3,
                            'asm3': asm3,

                            'correlation4': correlation4,
                            'contrast4': contrast4,
                            'homogeneity4': homogeneity4,
                            'energy4': energy4,
                            'asm4': asm4,

                            'correlation5': correlation5,
                            'contrast5': contrast5,
                            'homogeneity5': homogeneity5,
                            'energy5': energy5,
                            'asm5': asm5,

                            'correlation6': correlation6,
                            'contrast6': contrast6,
                            'homogeneity6': homogeneity6,
                            'energy6': energy6,
                            'asm6': asm6,

                            'class': int(folder_dir)})

            image_index += 1
        illness_index += 1
    return df_data


def get_nth_sum(df_data, i, n, key, d):

    if d != 0:
        low = 1 - d
        up = 1 + d
        SKO = random.choice(np.arange(low, up, 0.0001))
    else:
        SKO = 1
    if n == 1:
        return SKO * df_data[i][key]
    if n == 2:
        return SKO * (df_data[i-1][key] + df_data[i][key]) / n
    elif n == 3:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key]) / n
    elif n == 4:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key] + df_data[i+2][key]) / n
    elif n == 5:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key] + df_data[i+2][key] + df_data[i+3][key]) / n
    elif n == 6:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key] + df_data[i+2][key] + df_data[i+3][key] + df_data[i+4][key]) / n
    elif n == 7:
        return SKO * (
                    df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key]
                    + df_data[i + 3][key] + df_data[i + 4][key] + df_data[i + 5][key]) / n
    elif n == 8:
        return SKO * (
                df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key] + df_data[i + 3]
        [key] + df_data[i + 4][key] + df_data[i + 5][key] + + df_data[i + 6][key]) / n
    elif n == 9:
        return SKO * (
                df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key] + df_data[i + 3]
        [key] + df_data[i + 4][key] + df_data[i + 5][key] + df_data[i + 6][key] + df_data[i + 7][key]) / n
    elif n == 10:
        return SKO * (
                df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key] + df_data[i + 3]
        [key] + df_data[i + 4][key] + df_data[i + 5][key] + df_data[i + 6][key] + df_data[i + 7][key] + df_data[i + 8][
                    key]) / n


def get_hapalick_params(df_data_new, df_data, i, n, SKO):

    correlation1 = get_nth_sum(df_data, i, n, 'correlation1', SKO)
    contrast1 = get_nth_sum(df_data, i, n, 'contrast1', SKO)
    homogeneity1 = get_nth_sum(df_data, i, n, 'homogeneity1', SKO)
    energy1 = get_nth_sum(df_data, i, n, 'energy1', SKO)
    asm1 = get_nth_sum(df_data, i, n, 'asm1', SKO)

    correlation2 = get_nth_sum(df_data, i, n, 'correlation2', SKO)
    contrast2 = get_nth_sum(df_data, i, n, 'contrast2', SKO)
    homogeneity2 = get_nth_sum(df_data, i, n, 'homogeneity2', SKO)
    energy2 = get_nth_sum(df_data, i, n, 'energy2', SKO)
    asm2 = get_nth_sum(df_data, i, n, 'asm2', SKO)

    correlation3 = get_nth_sum(df_data, i, n, 'correlation3', SKO)
    contrast3 = get_nth_sum(df_data, i, n, 'contrast3', SKO)
    homogeneity3 = get_nth_sum(df_data, i, n, 'homogeneity3', SKO)
    energy3 = get_nth_sum(df_data, i, n, 'energy3', SKO)
    asm3 = get_nth_sum(df_data, i, n, 'asm3', SKO)

    correlation4 = get_nth_sum(df_data, i, n, 'correlation4', SKO)
    contrast4 = get_nth_sum(df_data, i, n, 'contrast4', SKO)
    homogeneity4 = get_nth_sum(df_data, i, n, 'homogeneity4', SKO)
    energy4 = get_nth_sum(df_data, i, n, 'energy4', SKO)
    asm4 = get_nth_sum(df_data, i, n, 'asm4', SKO)

    correlation5 = get_nth_sum(df_data, i, n, 'correlation5', SKO)
    contrast5 = get_nth_sum(df_data, i, n, 'contrast5', SKO)
    homogeneity5 = get_nth_sum(df_data, i, n, 'homogeneity5', SKO)
    energy5 = get_nth_sum(df_data, i, n, 'energy5', SKO)
    asm5 = get_nth_sum(df_data, i, n, 'asm5', SKO)

    correlation6 = get_nth_sum(df_data, i, n, 'correlation6', SKO)
    contrast6 = get_nth_sum(df_data, i, n, 'contrast6', SKO)
    homogeneity6 = get_nth_sum(df_data, i, n, 'homogeneity6', SKO)
    energy6 = get_nth_sum(df_data, i, n, 'energy6', SKO)
    asm6 = get_nth_sum(df_data, i, n, 'asm6', SKO)

    cl = df_data[i]['class']
    df_data_new.append({'correlation1': correlation1,
                        'contrast1': contrast1,
                        'homogeneity1': homogeneity1,
                        'energy1': energy1,
                        'asm1': asm1,

                        'correlation2': correlation2,
                        'contrast2': contrast2,
                        'homogeneity2': homogeneity2,
                        'energy2': energy2,
                        'asm2': asm2,

                        'correlation3': correlation3,
                        'contrast3': contrast3,
                        'homogeneity3': homogeneity3,
                        'energy3': energy3,
                        'asm3': asm3,

                        'correlation4': correlation4,
                        'contrast4': contrast4,
                        'homogeneity4': homogeneity4,
                        'energy4': energy4,
                        'asm4': asm4,

                        'correlation5': correlation5,
                        'contrast5': contrast5,
                        'homogeneity5': homogeneity5,
                        'energy5': energy5,
                        'asm5': asm5,

                        'correlation6': correlation6,
                        'contrast6': contrast6,
                        'homogeneity6': homogeneity6,
                        'energy6': energy6,
                        'asm6': asm6,

                        'class': cl
                        })
    return df_data_new


def generate_haralick_params(df_data, d, count, degrees):
    df_data_new = []
    # # d = 80
    # low = 1000 - d
    # up = 1000 + d
    for i in range(count):
        if 10 in degrees:
            for i in range(1, len(df_data) - 8):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class'] == \
                        df_data[i + 3]['class'] == df_data[i + 4]['class'] == df_data[i + 5]['class'] == df_data[i + 6]['class'] ==\
                        df_data[i + 7]['class'] == df_data[i + 8]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 10, d)
                else:
                    continue
        if 9 in degrees:
            for i in range(1, len(df_data) - 7):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class'] == \
                        df_data[i + 3]['class'] == df_data[i + 4]['class'] == df_data[i + 5]['class'] == df_data[i + 6]['class'] == df_data[i + 7]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 9, d)
                else:
                    continue
        if 8 in degrees:
            for i in range(1, len(df_data) - 6):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class'] == \
                        df_data[i + 3]['class'] == df_data[i + 4]['class'] == df_data[i + 5]['class'] == df_data[i + 6]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 8, d)
                else:
                    continue
        if 7 in degrees:
            for i in range(1, len(df_data) - 5):
                if df_data[i]['class'] == df_data[i-1]['class'] == df_data[i+1]['class'] == df_data[i+2]['class'] == df_data[i+3]['class'] == df_data[i+4]['class'] == df_data[i+5]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 7, d)
                else:
                    continue
        if 6 in degrees:
            for i in range(1, len(df_data) - 4):
                if df_data[i]['class'] == df_data[i-1]['class'] == df_data[i+1]['class'] == df_data[i+2]['class'] == df_data[i+3]['class'] == df_data[i+4]['class']:
                    # SKO = random.randint(low, up) / 1000
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 6, d)
                else:
                    continue
        if 5 in degrees:
            for i in range(1, len(df_data) - 3):
                if df_data[i]['class'] == df_data[i-1]['class'] == df_data[i+1]['class'] == df_data[i+2]['class'] == df_data[i+3]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 5, d)
                else:
                    continue
        if 4 in degrees:
            for i in range(1, len(df_data) - 2):
                if df_data[i]['class'] == df_data[i-1]['class'] == df_data[i+1]['class'] == df_data[i+2]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 4, d)
                else:
                    continue
        if 3 in degrees:
            for i in range(1, len(df_data)-1):
                if df_data[i]['class'] == df_data[i-1]['class'] == df_data[i+1]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 3, d)
                else:
                    continue
        if 2 in degrees:
            for i in range(1, len(df_data)):
                if df_data[i]['class'] == df_data[i - 1]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 2, d)
                else:
                    continue
        if 1 in degrees:
            for i in range(0, len(df_data)):
                df_data_new = get_hapalick_params(df_data_new, df_data, i, 1, d)

    return df_data_new


def get_optimal_params(x_train, y_train):
    parameter_space = {
        'alpha': [0.0001, 0.001, 0.01],
        'hidden_layer_sizes': [(30, 30), (40, 40), (50, 50), (100, 100), (100,)],
        'activation': ['logistic'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'max_iter': [600, 700, 800, 900, 1000, 1100, 1200]
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
    # accuracy = clf.score(x_test, y_test)
    # print("Accuracy = {}\n".format(accuracy))
    # target_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    # y_pred = clf.predict(x_test)
    # print(classification_report(y_test, y_pred, target_names=target_names))
    # disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    # print("Confusion matrix:\n%s" % disp.confusion_matrix)

    # return clf.best_params_


def get_optimal(x_train, y_train, x_test, y_test):
    parameter_space = {
        'alphas': [0.0001, 0.001, 0.01],
        'hidden_layer_sizes': [(30, 30), (40, 40), (50, 50), (100, 100), (100,)],
        'activations': ['relu', 'logistic'],
        'solvers': ['adam', 'lbfgs'],
        'max_iters': [600, 700, 800, 900, 1000, 1200]
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
            for activation in parameter_space['activations']:
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


                        print('{} <-- Activation = {} '
                              'Alpha = {} Hidden_layer_sizes = {}'
                              ' Solver = {} Epoch = {}'.format(accuracy, activation, alpha, hidden_layer_size, solver, max_iter))
    print('\nBest params:')
    print('{} <-- Activation = {} '
          'Alpha = {} Hidden_layer_sizes = {}'
          ' Solver = {} Epoch = {}'.format(best_accuracy, best_params['activation'],
                                           best_params['alpha'], best_params['hidden_layer_sizes'],
                                           best_params['solver'], best_params['max_iter']))


def get_acc_for_wheat():
    accs = []
    for i in range(10):
        df_train = (generator_data('./new_dir/train_dir/'))
        df_test = (generator_data('./new_dir/test_dir/'))
        # df_val = pd.DataFrame(generator_data('./val_dir/'))

        df_train = pd.DataFrame(generate_haralick_params(df_train, 0.1, 10, [4,3,2,1]))
        df_test = pd.DataFrame(generate_haralick_params(df_test, 0.0, 1, [1]))

        # df_test = pd.concat([df_test, df_val])
        x_test = df_test.iloc[:, 0:30]
        y_test = df_test.iloc[:, 30]

        # for i in range(12):
        #     numbers = df_test.iloc[0:10, i]
        #     plt.hist(numbers, bins=10, cumulative=False)
        #     plt.grid(linewidth=0.2)
        #     plt.title(i)
        #     plt.show()
        # #

        x_train = df_train.iloc[:, 0:30]
        y_train = df_train.iloc[:, 30]
        print('Train size = {}, test size = {}'.format(len(y_train), len(y_test)))

        clf = MLPClassifier(activation='relu',
                            max_iter=1200,
                            hidden_layer_sizes=(100,100),
                            solver='lbfgs',
                            early_stopping=True,
                            max_fun=20000,
                            verbose=0)

        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        accs.append(acc)
        print(acc)

        target_names = ['1', '2', '3', '4', '5', '6']
        y_pred = clf.predict(x_test)
        print(classification_report(y_test, y_pred, target_names=target_names))
        disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        print("Confusion matrix:\n%s" % disp.confusion_matrix)

    print(accs)
    print("MAX Accuracy = {}\nMIN Accuracy = {}\nAVG Accuracy = {}\n  ".format(max(accs), min(accs), mean(accs)))


def get_acc_for_eucalyptus():
    accs = []
    for i in range(10):
        df = pd.DataFrame(generator_data('./eucalyptus/'))
        df_train = (generator_data('./eucalyptus_train/'))
        df_test = (generator_data('./eucalyptus_test/'))
        # X = df.iloc[:, 0:12]
        # Y = df.iloc[:, 12]

        # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
        # get_optimal_params(X, Y)
        # df_test = (generator_data('./train_dir/'))
        # # df_val = pd.DataFrame(generator_data('./val_dir/'))

        df_train = pd.DataFrame(generate_haralick_params(df_train, 0.3, 50, [1,2,3]))
        df_test = pd.DataFrame(generate_haralick_params(df_test, 0.0, 1, [1]))

        # df_test = pd.concat([df_test, df_val])

        x_test = df_test.iloc[:, 0:30]
        y_test = df_test.iloc[:, 30]
        x_train = df_train.iloc[:, 0:30]
        y_train = df_train.iloc[:, 30]

        # for i in range(12):
        #     numbers = df_test.iloc[0:10, i]
        #     plt.hist(numbers, bins=10, cumulative=False)
        #     plt.grid(linewidth=0.2)
        #     plt.title(i)
        #     plt.show()

        print('Train size = {}, test size = {}'.format(len(y_train), len(y_test)))
        # get_optimal(x_train, y_train, x_test, y_test)
        clf = MLPClassifier(activation='logistic',
                            max_iter=1000,
                            hidden_layer_sizes=(50, 50, 50),
                            solver='adam',
                            early_stopping=True,
                            max_fun=20000,
                            verbose=0)

        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        accs.append(acc)
        print(acc)

        target_names = ['1', '2']
        y_pred = clf.predict(x_test)
        print(classification_report(y_test, y_pred, target_names=target_names))
        disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        print("Confusion matrix:\n%s" % disp.confusion_matrix)

    print(accs)
    print("MAX Accuracy = {}\nMIN Accuracy = {}\nAVG Accuracy = {}\n  ".format(max(accs), min(accs), mean(accs)))


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



    # get_acc_for_wheat()
    get_acc_for_eucalyptus()