import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.manifold import TSNE
import matplotlib.image as mpimg
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(tf.__version__)

DATA_LIST = os.listdir('two/train')
DATASET_PATH  = 'two/train'
TEST_DIR =  'two/test'
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = len(DATA_LIST)
BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 100 # Usually after 100 epochs with default settings, early stopping will kick in
LEARNING_RATE = 0.0001 # start off with high rate first 0.001 and experiment with reducing it gradually 

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=50,featurewise_center = True, featurewise_std_normalization = True, width_shift_range=0.2, height_shift_range=0.2,shear_range=0.25,zoom_range=0.1, zca_whitening = True,channel_shift_range = 20, horizontal_flip = True, vertical_flip = True, validation_split = 0.2,fill_mode='constant')

train_batches = train_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE, shuffle=True,batch_size=BATCH_SIZE, subset = "training", seed=42, class_mode="binary")

valid_batches = train_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE, shuffle=True,batch_size=BATCH_SIZE, subset = "validation", seed=42, class_mode="binary")

xception = tf.keras.applications.xception.Xception(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
model3=tf.keras.Sequential()
model3.add(xception)
model3.add(tf.keras.layers.Flatten())
model3.add(tf.keras.layers.Dense(2, activation='softmax'))

model3.summary()
model3.layers[0].summary()

#FIT MODEL
print(len(train_batches))
print(len(valid_batches))

model3.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics = ['accuracy'])

early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=40, verbose=1, mode='auto')

STEP_SIZE_TRAIN=train_batches.n//train_batches.batch_size
STEP_SIZE_VALID=valid_batches.n//valid_batches.batch_size

history = model3.fit(train_batches, steps_per_epoch = STEP_SIZE_TRAIN, epochs = NUM_EPOCHS, validation_data = valid_batches, validation_steps = STEP_SIZE_VALID, callbacks=[early])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.9, 1])
plt.legend(loc='lower right')

test_datagen = ImageDataGenerator(rescale=1. / 255)
eval_generator = test_datagen.flow_from_directory(TEST_DIR,target_size=IMAGE_SIZE, batch_size=1,shuffle=False,seed=42,class_mode="binary")

eval_generator.reset()
pred = model3.predict(eval_generator,18,verbose=1)
for index, probability in enumerate(pred):
    image_path = TEST_DIR + "/" +eval_generator.filenames[index]
    image = mpimg.imread(image_path)
    if image.ndim < 3:
        image = np.reshape(image,(image.shape[0],image.shape[1],1))
        image = np.concatenate([image, image, image], 2)
#         print(image.shape)

    pixels = np.array(image)
    plt.imshow(pixels)
#     print(index, probability)
    print(eval_generator.filenames[index])
    if probability[0] < 0.5:
        plt.title("%.2f" % (probability[1]*100) + "% Normal (" + "%.8f" % (probability[1]) + ")")
    else:
        plt.title("%.2f" % ((probability[0])*100) + "% COVID19 Pneumonia (" + "%.8f" % (probability[0]) + ")")
    plt.show()

intermediate_layer_model = tf.keras.models.Model(inputs=model3.input, outputs=model3.get_layer('dense').output)
tsne_data_generator = test_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE, batch_size=1,shuffle=False,seed=42,class_mode="binary")

data = intermediate_layer_model.predict(tsne_data_generator)

embedded_data = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data)

colors = ['#8bbe1b', '#ff77b4']
C = [colors[i] for i in tsne_data_generator.classes]

c_dict = {0: '#8bbe1b', 1: '#ff77b4'}
group = tsne_data_generator.classes
labels = tsne_data_generator.class_indices

l = {v: k for k, v in labels.items()}
fig, ax = plt.subplots()

for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(embedded_data[ix, 0], embedded_data[ix, 1], c = c_dict[g], label = l[g])

ax.legend()
plt.show()