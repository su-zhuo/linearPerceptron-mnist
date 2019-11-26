import pandas as pd
import numpy as np
import cv2
import random
import time
import readMNIST

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


def Train(train_data, train_labels, class0):
    train_data_size = len(train_labels)
    w = np.zeros((feature_length, 1))
    b = 0
    study_count = 0
    nochange_count = 0  #
    nochange_upper_limit = 100000
    while True:
        nochange_count += 1
        if nochange_count > nochange_upper_limit:
            break

        index = random.randint(0, train_data_size - 1)
        img = train_data[index]
        label = train_labels[index]
        yi = int(label != class0) * 2 - 1
        result = yi * (np.dot(img, w) + b)

        if result <= 0:
            img = np.reshape(train_data[index], (feature_length, 1))
            w += img * yi * study_step
            b += yi * study_step

            study_count += 1
            if study_count > study_total:
                break
            nochange_count = 0

    return w, b


def Test(test_data, w, b, class0, class1):
    predict = []
    for img in test_data:
        result = np.dot(img, w) + b
        # result = result > 0
        if result < 0:
            result = class0
        else:
            result = class1

        predict.append(result)

    return np.array(predict)

def Hog(trainset):
    features = []

    hog = cv2.HOGDescriptor('./hog.xml')

    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return features


study_step = 0.0001
study_total = 10000
feature_length = 324
from sklearn.decomposition import PCA
if __name__ == '__main__':
    acc = 0
    time_train = 0
    time_test = 0
    for class0 in range(10):
        for class1 in range(class0 + 1, 10):
            print("let's use :", class0, "and", class1)
            time_1 = time.time()

            train_imgs = readMNIST.loadImageSet('train')
            train_labels = readMNIST.loadLabelSet('train')
            test_imgs = readMNIST.loadImageSet('test')
            test_labels = readMNIST.loadLabelSet('test')

            train_imgs = train_imgs[np.squeeze(np.argwhere((train_labels == class0) | (train_labels == class1)))[:, 0],:]
            train_labels = train_labels[(train_labels == class0) | (train_labels == class1)]
            test_imgs = test_imgs[np.squeeze(np.argwhere((test_labels == class0) | (test_labels == class1)))[:, 0], :]
            test_labels = test_labels[(test_labels == class0) | (test_labels == class1)]

            # n_components = 100
            # pca = PCA(n_components=n_components).fit(train_imgs)
            # train_imgs = pca.transform(train_imgs)
            # test_imgs = pca.transform(test_imgs)
            #
            # train_features = train_imgs
            # test_features = test_imgs
            train_features = Hog(train_imgs)
            test_features = Hog(test_imgs)

            time_2 = time.time()
            w, b = Train(train_features, train_labels, class0)
            time_3 = time.time()

            test_predict = Test(test_features, w, b, class0, class1)
            time_4 = time.time()

            score = accuracy_score(test_labels, test_predict)
            acc += score
            time_train += (time_3 - time_2)
            time_test += (time_4 - time_3)

            print("The accuracy score of class ", class0, " and ", class1, " is ", score)

    print("The accuracy score in average is ", acc / 45)
    print("The time_train in average is ", time_train / 45)
    print("The time_test in average is ", time_test / 45)
