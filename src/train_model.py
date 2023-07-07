import os
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer

# config

CSV_FILE = os.path.abspath('data/movies.csv')
IMG_FOLDER = os.path.abspath('data/Multi_Label_dataset/Images')

TOTAL_EPOCHS = 100
LEARNING_RATE = 0.1
BATCH_SIZE = 64
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


def build_model(input_shape, num_classes):
	# model = tf.keras.models.Sequential([
	# tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
	# tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
	# tf.keras.layers.MaxPooling2D(2,2),
	# tf.keras.layers.Dropout(0.25),
	
	# tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
	# tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
	# tf.keras.layers.MaxPooling2D(2,2),
	# tf.keras.layers.Dropout(0.25),
	
	# tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
	# tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
	# tf.keras.layers.MaxPooling2D(2,2),
	# tf.keras.layers.Dropout(0.25),
	
	# tf.keras.layers.Flatten(),
	# tf.keras.layers.Dense(1024, activation='relu'),
	# tf.keras.layers.Dropout(0.5),
	# tf.keras.layers.Dense(num_classes, activation='sigmoid')])
	model = tf.keras.models.Sequential()
	model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dense(num_classes, activation='sigmoid'))
	
	model.compile(loss='binary_crossentropy',
		optimizer=Adam(LEARNING_RATE),
		metrics=['acc'])
	
	return model


# main script

images, labels = load_data(CSV_FILE, IMG_FOLDER)

MLB = MultiLabelBinarizer()
LABELS = MLB.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, LABELS, test_size=0.2, random_state=42)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_it = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
validate_it = datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)

if not os.path.exists('models'):
	os.makedirs('models')

model = build_model(X_train.shape[1:], len(MLB.classes_))

history = model.fit(
	train_it,
	steps_per_epoch = STEPS_PER_EPOCH,
	epochs = TOTAL_EPOCHS,
	validation_data = validate_it,
	validation_steps = STEPS_PER_EPOCH
)
name = f'epochs{TOTAL_EPOCHS}_lr{LEARNING_RATE}_batch{BATCH_SIZE}_steps{STEPS_PER_EPOCH}'
model.save(f'models/{name}-model.h5')