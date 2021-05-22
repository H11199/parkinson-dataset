from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

# This is how labels is encoded
'''
label = []
label = "parkinson-dataset/spiral/training/healthy/".split("/")[-2]
print(label)
'''
# output = healthy

def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    # below is how imagePaths list look like
    #['parkinson-dataset/spiral/training\\healthy\\V01HE02.png','parkinson-dataset/spiral/training\\healthy\\V01HE03.png',...]


    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        # [healthy, healthy, parkinson, ....]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features = quantify_image(image)
        data.append(features)
        labels.append(label)
    '''
    below is how data array look like
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
 ...
     [0. 0. 0. ... 0. 0. 0.]]

    '''

    '''
    Below is how labels array look like
    ['healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy'
     'healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy'
     'healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy'
     'healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy'
     'healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy' 'healthy']
    '''

    return (np.array(data), np.array(labels))


trainingPath = "parkinson-dataset/spiral/training"
testingPath = "parkinson-dataset/spiral/testing"
# loading the training and testing data
print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)
# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

trials = {}

for i in range(0, 5):
    print("[INFO] training model {} of {}...".format(i + 1,5))
    model = RandomForestClassifier(n_estimators=100)
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    metrics = {}
    cm = confusion_matrix(testY, predictions).flatten()
    (tn, fp, fn, tp) = cm
    metrics["acc"] = (tp + tn) / float(cm.sum())
    metrics["sensitivity"] = tp / float(tp + fn)
    metrics["specificity"] = tn / float(tn + fp)
    for (k, v) in metrics.items():
        l = trials.get(k, [])
        l.append(v)
        trials[k] = l

    # below how we get trials dictionary with 3 keys
    # {'new': [23.445, 25.345, 23.443,...], 'newone': [28.445, 28.445,22.999,...]}

for metric in ("acc", "sensitivity", "specificity"):
    # grab the list of values for the current metric, then compute
    # the mean and standard deviation
    values = trials[metric]
    mean = np.mean(values)
    std = np.std(values)
    # show the computed metrics for the statistic
    print(metric)
    print("=" * len(metric))
    print("u={:.4f}, o={:.4f}".format(mean, std))
    print("")

'''
acc
===
u=0.8200, o=0.0452

sensitivity
===========
u=0.7600, o=0.0533

specificity
===========
u=0.8800, o=0.0499
'''
testingPaths = list(paths.list_images(testingPath))
idxs = np.arange(0, len(testingPaths))
idxs = np.random.choice(idxs, size=(25,), replace=False)
images = []
# loop over the testing samples
for i in idxs:
    # load the testing image, clone it, and resize it
    image = cv2.imread(testingPaths[i])
    output = image.copy()
    output = cv2.resize(output, (128, 128))
    # pre-process the image in the same manner we did earlier
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    preds = model.predict([features])
    label = le.inverse_transform(preds)[0]
    # draw the colored class label on the output image and add it to
    # the set of output images
    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
    images.append(output)

# create a montage using 128x128 "tiles" with 5 rows and 5 columns

montage = build_montages(images, (128, 128), (5, 5))[0]
# show the output montage
cv2.imshow("Output", montage)
cv2.waitKey(0)



def classify_my_image(image_path):
    image = cv2.imread(image_path)
    output = image.copy()
    output = cv2.resize(output, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    preds = model.predict([features])


    #print(preds)
    # preds gives list of encoded label 1 -> parkinson and 0 -> healthy here list is [1] <-> [parkinson]
    #print(le.inverse_transform(preds))


    label = le.inverse_transform(preds)[0]

    # le.inverse_transform(preds) convert [1] to [parkinson] and le.inverse_transform(preds)[0] -> parkinson

    color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output montage
    plt.imshow(output)
    plt.show()
    cv2.waitKey(0)

'''
imageUpload = "parkinson-dataset/drawings/spiral/testing/parkinson/V02PE01.png"
classify_my_image(imageUpload)
'''