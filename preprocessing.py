import cv2, os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np

learning_rate = 1e-4
EPOCHS = 10

path = 'dataset'
CATEGORIES = ['mask','no_mask']

data = []
labels = []

for category in CATEGORIES:
    category_path = os.path.join(path, category)
    for img in os.listdir(category_path):
        img_path = os.path.join(category_path, img)
        image = load_img(img_path, target_size=(100,100))
        image = img_to_array(image)
        # image = preprocess_input(image)

        data.append(image)
        labels.append(category)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

print(data)
print(labels)

np.save('data',data)
np.save('labels',labels)


