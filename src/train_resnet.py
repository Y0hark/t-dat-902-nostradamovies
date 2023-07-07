import tensorflow as tf
import os
import pandas as pd
import cv2
import numpy as np

from keras.layers import Input, Conv2D, Add, ReLU, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# config

CSV_FILE = os.path.abspath('data/movies.csv')
IMG_FOLDER = os.path.abspath('data/Multi_Label_dataset/Images')

TOTAL_EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 10

def load_data(csv_file, img_folder, target_size=(224, 224)):
	df = pd.read_csv(csv_file)
	images = []
	labels = []
	for _, row in df.iterrows():
		img_file = os.path.join(img_folder, row['imdb_title_id'] + '.jpg')
		if os.path.exists(img_file):
			img = cv2.imread(img_file)
			img = cv2.resize(img, target_size)
			images.append(img)
			labels.append(row['genre'].split(','))
	return np.array(images), labels

def residual_block(x, filters, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size=3, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    if stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])
    x = ReLU()(x)

    return x

def create_resnet(input_shape, num_classes, num_residual_blocks=4):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=7, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    num_filters = 64
    for i in range(num_residual_blocks):
        x = residual_block(x, filters=num_filters, stride=2 if i > 0 else 1)
        num_filters *= 2

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(LEARNING_RATE), metrics=['acc'])

    return model

# Load data
images, labels = load_data(CSV_FILE, IMG_FOLDER)
images = images / 255.0  # Normalize images

MLB = MultiLabelBinarizer()
LABELS = MLB.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, LABELS, test_size=0.2, random_state=42)

datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train_it = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
validate_it = datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)

if not os.path.exists('models'):
	os.makedirs('models')

# Créez le modèle
input_shape = (X_train.shape[1:])  # Remplacez par les dimensions de vos images
num_classes = len(MLB.classes_)  # Remplacez par le nombre de classes que vous avez
model = create_resnet(input_shape, num_classes)

# Entraînez le modèle
history = model.fit(train_it,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=TOTAL_EPOCHS,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=(validate_it))
