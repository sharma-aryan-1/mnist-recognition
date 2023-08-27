import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os


# import and load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# pre-processing
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)


# testing neural network

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)


# testing pt. 2

imgno = 1
while os.path.isfile(f"digits/d{imgno}.png"):
    try:
        img = cv2.imread(f"digits/d{imgno}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        imgno+=1
