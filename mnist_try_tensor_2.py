import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

some_random_digit = cv2.imread("images/test_2.jpg" ,cv2.IMREAD_GRAYSCALE)
some_random_digit = cv2.resize(some_random_digit, (28, 28))
some_random_digit = tf.keras.utils.normalize(some_random_digit, axis=1)
some_random_digit = cv2.resize(some_random_digit, (28, 28))
some_random_digit_2 = []
some_random_digit_2.append(some_random_digit)
for data in x_test:
    some_random_digit_2.append(data)

some_random_digit_3 = []
some_random_digit_3.append(2)
for features in y_test:
    some_random_digit_3.append(features)

some_random_digit_2 = np.array(some_random_digit_2).reshape(-1, 28, 28)
plt.imshow(some_random_digit_2[0], cmap='gray') 
plt.show()

# model = tf.keras.models.load_model('epic_num_reader_4.h5')

model = tf.keras.models.Sequential()  
model.add(tf.keras.layers.Flatten(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=7)

val_loss, val_acc = model.evaluate(some_random_digit_2, some_random_digit_3)
print(val_loss)
print(val_acc)

predictions = model.predict(some_random_digit_2)
print(np.argmax(predictions[0]))
plt.imshow(some_random_digit_2[0], cmap='gray') 
plt.show()