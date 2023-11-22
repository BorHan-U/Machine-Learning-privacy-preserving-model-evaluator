#import Packages
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import PIL
from PIL import ImageEnhance, ImageOps, Image
from matplotlib import pyplot
from keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

#default Plot sizes
plt.rcParams["figure.figsize"] = (16, 10)  # Make the plots bigger by default
plt.rcParams["lines.linewidth"] = 2  # Setting the default line width
plt.style.use("ggplot")

#Directory set up and image size
data_dir = '/Users/borhan/Desktop/Data Science/project/code'
train_path = '/Users/borhan/Desktop/Data Science/project/code/Train'
test_path = '/Users/borhan/Desktop/Data Science/project/code'
IMG_HEIGHT = 30
IMG_WIDTH = 30

# Number of Classes
NUM_CATEGORIES = len(os.listdir(train_path))
NUM_CATEGORIES

# Visualizing all the different Signs
img_dir = pathlib.Path(train_path)
plt.figure(figsize=(14, 14))
index = 0
for i in range(NUM_CATEGORIES):
    plt.subplot(7, 7, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    sign = list(img_dir.glob(f'{i}/*'))[0]
    img = load_img(sign, target_size=(IMG_WIDTH, IMG_HEIGHT))
    plt.imshow(img)
plt.show()

# Label Overview
classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}


folders = os.listdir(train_path)

#Labal of class overview visualization
train_number = []
class_num = []

for folder in folders:
    train_files = os.listdir(train_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[int(folder)])


plt.figure(figsize=(21, 10))
plt.bar(class_num, train_number, color='green')
plt.xticks(class_num, rotation='vertical')
plt.show()

# Load data from directory including classes
def load_data(data_dir):

    images = list()
    labels = list()
    for category in range(NUM_CATEGORIES):
        categories = os.path.join(data_dir, str(category))

    for img in os.listdir(categories):
        img = load_img(os.path.join(categories, img), target_size=(30, 30))
        image = img_to_array(img)
        images.append(image)
        labels.append(category)

    return images, labels

images, labels = load_data(train_path)
labels = to_categorical(labels)


#Train & Test the data
x_train, x_test, y_train, y_test = train_test_split(
    np.array(images), labels, test_size=0.3)

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

print('x_train shape:', x_test.shape)



#Modelling
input_shape = (30, 30, 3)
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=3,
                 activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Second Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Third Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(NUM_CATEGORIES, activation='softmax'))
# Compiling the model
lr = 0.001
epochs = 30
model.compile(loss='categorical_crossentropy',
              optimizer="adam", metrics=['accuracy'])

model.summary()


history = model.fit(x_train, y_train, validation_split=0.3, epochs=20)

loss, accuracy = model.evaluate(x_test, y_test)

print('test set accuracy: ', accuracy * 100)

plt.plot(history.history['accuracy'], color='green')
plt.plot(history.history['val_accuracy'], color='grey')
plt.title('Acccuracy_Of_Model')
plt.ylabel('accuracy')
plt.xlabel('Per_epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

for i in range(36):
    pyplot.subplot(6, 6, i+1)
    pyplot.imshow(x_test[i])


image_index = 0
plt.imshow(x_test[image_index])
n = np.array(x_test[image_index])
print(n.size)
p = n.reshape(1, 30, 30, 3)
pred = classes[model.predict(p).argmax()]

print("The predicted image is {}".format(pred))
