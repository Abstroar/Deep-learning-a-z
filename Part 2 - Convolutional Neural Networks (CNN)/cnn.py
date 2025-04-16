import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from keras.preprocessing import image

def pre(path):
    s = image.load_img(path, target_size = (64,64))
    s = image.img_to_array(s)
    s = np.expand_dims(s, axis = 0)
    return s

#preprocessing
if __name__ == "__main__":
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    train_set = train_datagen.flow_from_directory(
        './dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255,
    )
    test_set = test_datagen.flow_from_directory(
        './dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary'
    )
    # building the cnn
    cnn = Sequential()
    cnn.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu',input_shape=(64,64,3)))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D())
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(32, activation='relu'))
    cnn.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    cnn.compile(optimizer='adam' ,loss='binary_crossentropy', metrics= ['accuracy'])

    cnn.fit(x = train_set, validation_data=test_set, epochs=26)


    pred_1 = cnn.predict(pre('dataset/single_prediction/cat_or_dog_1.jpg'))
    pred_2 = cnn.predict(pre('dataset/single_prediction/cat_or_dog_2.jpg'))
    print("predicting images",pred_1,pred_2)
    print(train_set.class_indices)