from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = '/Users/bharathmahesh/Downloads/archive (1)/train'
valid_path = '/Users/bharathmahesh/Downloads/archive (1)/test'

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = '/Users/bharathmahesh/Downloads/archive (1)/train'
valid_path = '/Users/bharathmahesh/Downloads/archive (1)/test'

vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg16.layers:
    layer.trainable = False

folders = glob('/Users/bharathmahesh/Downloads/archive (1)/train/*')
x = Flatten()(vgg16.output)

prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg16.input, outputs=prediction)
#%%
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
#%%
# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/Users/bharathmahesh/Downloads/archive (1)/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/Users/bharathmahesh/Downloads/archive (1)/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=4,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('/Users/bharathmahesh/Downloads/archive (1)/test/Healthy/zoom_35.jpg', target_size = (224,224))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


result
a= np.argmax(model.predict(test_image),axis=1)
a
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

IMAGE_SIZE = [224, 224]

train_path1 = '/Users/bharathmahesh/Downloads/archive (1)/train'
valid_path1 = '/Users/bharathmahesh/Downloads/archive (1)/test'

ResNet50 = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in ResNet50.layers:
    layer.trainable = False

folders1 = glob('/Users/bharathmahesh/Downloads/archive (1)/train/*')
x1 = Flatten()(ResNet50.output)

prediction1 = Dense(len(folders1), activation='softmax')(x1)

# create a model object
model1 = Model(inputs=ResNet50.input, outputs=prediction1)
model1.summary()

model1.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen1 = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen1 = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen1.flow_from_directory('/Users/bharathmahesh/Downloads/archive (1)/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen1.flow_from_directory('/Users/bharathmahesh/Downloads/archive (1)/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
d = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=4,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('/Users/bharathmahesh/Downloads/archive (1)/test/Healthy/translation_zoom_9.jpg', target_size = (224,224))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result1 = model1.predict(test_image)

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

result1
result1 = result1.ravel()
result1=result1.tolist()

result1
a1= np.argmax(model1.predict(test_image),axis=1)
a1

classes = ["diseased_leaf", "diseased_plant", "fresh_leaf", "fresh_plant"]
max = result1[0];
i = 0;

# Loop through the array
for index, value in enumerate(result1):
    # Compare elements of array with max
    if (value > max):
        max = value;
        i = index
print("Largest element present in given array: " + str(max) + " And it belongs to " + str(classes[i])+" class.");
