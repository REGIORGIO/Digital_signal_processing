import pandas as pd
from skimage.feature import greycomatrix, greycoprops
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import random
import cv2 as cv
from statistics import mean
from matplotlib import pyplot as plt


def convert_images_to_params(root, file_to_save_name='input_params.csv'):
    image_index = 0
    illness_index = 1
    masks = [[132, 0, 0], # red
             [204, 208, 193], # grey
             [245, 228, 103], # light yellow
             [54, 46, 34], # brown
             [182, 146, 82], # light brown
             [0, 255, 0]] # light green
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
            # print(path)
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
            dissimilarity1 = greycoprops(glcms[0], 'dissimilarity')[0, 0]

            correlation2 = greycoprops(glcms[1], 'correlation')[0, 0]
            contrast2 = greycoprops(glcms[1], 'contrast')[0, 0]
            homogeneity2 = greycoprops(glcms[1], 'homogeneity')[0, 0]
            energy2 = greycoprops(glcms[1], 'energy')[0, 0]
            dissimilarity2 = greycoprops(glcms[1], 'dissimilarity')[0, 0]

            correlation3 = greycoprops(glcms[2], 'correlation')[0, 0]
            contrast3 = greycoprops(glcms[2], 'contrast')[0, 0]
            homogeneity3 = greycoprops(glcms[2], 'homogeneity')[0, 0]
            energy3 = greycoprops(glcms[2], 'energy')[0, 0]
            dissimilarity3 = greycoprops(glcms[2], 'dissimilarity')[0, 0]

            correlation4 = greycoprops(glcms[3], 'correlation')[0, 0]
            contrast4 = greycoprops(glcms[3], 'contrast')[0, 0]
            homogeneity4 = greycoprops(glcms[3], 'homogeneity')[0, 0]
            energy4 = greycoprops(glcms[3], 'energy')[0, 0]
            dissimilarity4 = greycoprops(glcms[3], 'dissimilarity')[0, 0]

            correlation5 = greycoprops(glcms[4], 'correlation')[0, 0]
            contrast5 = greycoprops(glcms[4], 'contrast')[0, 0]
            homogeneity5 = greycoprops(glcms[4], 'homogeneity')[0, 0]
            energy5 = greycoprops(glcms[4], 'energy')[0, 0]
            dissimilarity5 = greycoprops(glcms[4], 'dissimilarity')[0, 0]

            correlation6 = greycoprops(glcms[5], 'correlation')[0, 0]
            contrast6 = greycoprops(glcms[5], 'contrast')[0, 0]
            homogeneity6 = greycoprops(glcms[5], 'homogeneity')[0, 0]
            energy6 = greycoprops(glcms[5], 'energy')[0, 0]
            dissimilarity6 = greycoprops(glcms[5], 'dissimilarity')[0, 0]

            df_data.append({'correlation1': correlation1,
                            'contrast1': contrast1,
                            'homogeneity1': homogeneity1,
                            'energy1': energy1,
                            'dissimilarity1': dissimilarity1,

                            'correlation2': correlation2,
                            'contrast2': contrast2,
                            'homogeneity2': homogeneity2,
                            'energy2': energy2,
                            'dissimilarity2': dissimilarity2,

                            'correlation3': correlation3,
                            'contrast3': contrast3,
                            'homogeneity3': homogeneity3,
                            'energy3': energy3,
                            'dissimilarity3': dissimilarity3,

                            'correlation4': correlation4,
                            'contrast4': contrast4,
                            'homogeneity4': homogeneity4,
                            'energy4': energy4,
                            'dissimilarity4': dissimilarity4,

                            'correlation5': correlation5,
                            'contrast5': contrast5,
                            'homogeneity5': homogeneity5,
                            'energy5': energy5,
                            'dissimilarity5': dissimilarity5,

                            'correlation6': correlation6,
                            'contrast6': contrast6,
                            'homogeneity6': homogeneity6,
                            'energy6': energy6,
                            'dissimilarity6': dissimilarity6,

                            'class': int(folder_dir)})

            image_index += 1
        illness_index += 1

    pd.DataFrame(df_data).to_csv(file_to_save_name, sep='\t', index=False)
    # return df_data


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
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key] + df_data[i+2][key] + df_data[i+3][key] +
                      df_data[i+4][key]) / n
    elif n == 7:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key] + df_data[i+2][key] + df_data[i+3][key] +
                      df_data[i+4][key] + df_data[i+5][key]) / n
    elif n == 8:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key] + df_data[i+2][key] + df_data[i+3][key] +
                      df_data[i+4][key] + df_data[i+5][key] + df_data[i+6][key]) / n
    elif n == 9:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key] + df_data[i+2][key] + df_data[i+3][key] +
                      df_data[i+4][key] + df_data[i+5][key] + df_data[i+6][key] + df_data[i+7][key]) / n
    elif n == 10:
        return SKO * (df_data[i-1][key] + df_data[i][key] + df_data[i+1][key] + df_data[i+2][key] + df_data[i+3][key] +
                      df_data[i+4][key] + df_data[i+5][key] + df_data[i+6][key] + df_data[i+7][key] + df_data[i+8][key]) / n


def get_hapalick_params(df_data_new, df_data, i, n, SKO):

    correlation1 = get_nth_sum(df_data, i, n, 'correlation1', SKO)
    contrast1 = get_nth_sum(df_data, i, n, 'contrast1', SKO)
    homogeneity1 = get_nth_sum(df_data, i, n, 'homogeneity1', SKO)
    energy1 = get_nth_sum(df_data, i, n, 'energy1', SKO)
    dissimilarity1 = get_nth_sum(df_data, i, n, 'dissimilarity1', SKO)

    correlation2 = get_nth_sum(df_data, i, n, 'correlation2', SKO)
    contrast2 = get_nth_sum(df_data, i, n, 'contrast2', SKO)
    homogeneity2 = get_nth_sum(df_data, i, n, 'homogeneity2', SKO)
    energy2 = get_nth_sum(df_data, i, n, 'energy2', SKO)
    dissimilarity2 = get_nth_sum(df_data, i, n, 'dissimilarity2', SKO)

    correlation3 = get_nth_sum(df_data, i, n, 'correlation3', SKO)
    contrast3 = get_nth_sum(df_data, i, n, 'contrast3', SKO)
    homogeneity3 = get_nth_sum(df_data, i, n, 'homogeneity3', SKO)
    energy3 = get_nth_sum(df_data, i, n, 'energy3', SKO)
    dissimilarity3 = get_nth_sum(df_data, i, n, 'dissimilarity3', SKO)

    correlation4 = get_nth_sum(df_data, i, n, 'correlation4', SKO)
    contrast4 = get_nth_sum(df_data, i, n, 'contrast4', SKO)
    homogeneity4 = get_nth_sum(df_data, i, n, 'homogeneity4', SKO)
    energy4 = get_nth_sum(df_data, i, n, 'energy4', SKO)
    dissimilarity4 = get_nth_sum(df_data, i, n, 'dissimilarity4', SKO)

    correlation5 = get_nth_sum(df_data, i, n, 'correlation5', SKO)
    contrast5 = get_nth_sum(df_data, i, n, 'contrast5', SKO)
    homogeneity5 = get_nth_sum(df_data, i, n, 'homogeneity5', SKO)
    energy5 = get_nth_sum(df_data, i, n, 'energy5', SKO)
    dissimilarity5 = get_nth_sum(df_data, i, n, 'dissimilarity5', SKO)

    correlation6 = get_nth_sum(df_data, i, n, 'correlation6', SKO)
    contrast6 = get_nth_sum(df_data, i, n, 'contrast6', SKO)
    homogeneity6 = get_nth_sum(df_data, i, n, 'homogeneity6', SKO)
    energy6 = get_nth_sum(df_data, i, n, 'energy6', SKO)
    dissimilarity6 = get_nth_sum(df_data, i, n, 'dissimilarity6', SKO)

    cl = df_data[i]['class']
    df_data_new.append({'correlation1': correlation1,
                        'contrast1': contrast1,
                        'homogeneity1': homogeneity1,
                        'energy1': energy1,
                        'dissimilarity1': dissimilarity1,

                        'correlation2': correlation2,
                        'contrast2': contrast2,
                        'homogeneity2': homogeneity2,
                        'energy2': energy2,
                        'dissimilarity2': dissimilarity2,

                        'correlation3': correlation3,
                        'contrast3': contrast3,
                        'homogeneity3': homogeneity3,
                        'energy3': energy3,
                        'dissimilarity3': dissimilarity3,

                        'correlation4': correlation4,
                        'contrast4': contrast4,
                        'homogeneity4': homogeneity4,
                        'energy4': energy4,
                        'dissimilarity4': dissimilarity4,

                        'correlation5': correlation5,
                        'contrast5': contrast5,
                        'homogeneity5': homogeneity5,
                        'energy5': energy5,
                        'dissimilarity5': dissimilarity5,

                        'correlation6': correlation6,
                        'contrast6': contrast6,
                        'homogeneity6': homogeneity6,
                        'energy6': energy6,
                        'dissimilarity6': dissimilarity6,

                        'class': cl
                        })

    return df_data_new


def generate_haralick_params(df_data, d, count, degrees):
    df_data_new = []

    for k in range(count):

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

    return pd.DataFrame(df_data_new)


def get_classification_report(data_train, data_test, class_names):
    x_train = data_train.iloc[:, 0:30]
    y_train = data_train.iloc[:, 30]

    x_test = data_test.iloc[:, 0:30]
    y_test = data_test.iloc[:, 30]

    print('Train size = {}, test size = {}'.format(len(y_train), len(y_test)))

    clf = MLPClassifier(activation='logistic',
                        max_iter=1200,
                        hidden_layer_sizes=(100, 100),
                        solver='lbfgs',
                        early_stopping=True,
                        random_state=1)

    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print(acc)

    # show confusion matrix
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=class_names))
    disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.close()

    return acc


def get_statistics(accs):
    print('\nResults')
    print(accs)
    print("MAX Accuracy = {}\nMIN Accuracy = {}\nAVG Accuracy = {}\n  ".format(max(accs), min(accs), mean(accs)))
    plt.hist(accs, bins=10, cumulative=False)
    plt.grid(linewidth=0.2)
    plt.show()


def get_accuracy(class_names, root_folder, stat_count, input_train_params, input_test_params, SKO=0.05, iteration_count=100, averaged_elements=[10]):
    accs = []

    for i in range(stat_count):
        print("Iteration # {}".format(i + 1))

        with open(input_train_params) as train, open(input_test_params) as test:
            data_train = pd.read_csv(train, sep='\t')
            generated_data_train = generate_haralick_params(data_train.T.to_dict(), SKO, iteration_count, averaged_elements)
            # print(generated_data_train)

            data_test = pd.read_csv(test, sep='\t')

            accs.append(get_classification_report(generated_data_train, data_test, class_names))

            if accs[len(accs) - 1] == max(accs):
                generated_data_train.to_csv(root_folder + 'best_generated_params.csv', sep='\t', index=False)

    # show statistic results
    if stat_count > 1:
        get_statistics(accs)
