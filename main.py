import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import pandas

batch_size = 32
img_height = 180
img_width = 180

#splitting and training
train_ds = tf.keras.utils.image_dataset_from_directory(
"D:\DTL_Project\cocoon_images",
validation_split=0.2,
subset="training",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)


#splitting and validation
val_ds = tf.keras.utils.image_dataset_from_directory(
  "D:\DTL_Project\cocoon_images",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

df = pandas.read_excel('cocoons_weight.xlsx')
df.info
print(df.info)

class_names = train_ds.class_names
print(class_names)

#TUNING THE DATA MODEL
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#SPLITTING THE DATA INTO 0 AND 1s
normalization_layer = layers.Rescaling(1./255)
#APPLYING TO ALL IMAGES
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

#CREATING A MODEL
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#COMPILATION OF MODEL
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary() #SUMMARY OF THE MODEL

#FITTING THE MODEL
epochs=10 
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#CHECKING THE ACCURACY OF THE MODEL 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

cocoon_url = "D:\DTL_Project\test.jpeg"
#cocoon_path = tf.keras.utils.get_file('test', origin=cocoon_url)

# img = tf.keras.utils.load_img(
#     cocoon_url, target_size=(img_height, img_width)
# )
img = tf.keras.utils.load_img(
    "test.jpeg",target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

#SIMPLE IF ELSE STATEMENT FOR MALE OR FEMALE.
dict_cocoon = {"pure_mysore_silk":[0.8,1.1,1.2,1.4],"tasar":[2.0,2.49,2.5,3.5]}
cocoon_breed = class_names[np.argmax(score)]
if(cocoon_breed in dict_cocoon.keys()):
  weight = 2.3
  if(weight>dict_cocoon[cocoon_breed][0] and weight<dict_cocoon[cocoon_breed][1]):
    print("Male")
  elif(weight>dict_cocoon[cocoon_breed][1] and weight<dict_cocoon[cocoon_breed][2]):
    print("Female")