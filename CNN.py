Definitions
from google.colab import drive
drive.mount('/content/drive')
 %matplotlib inline
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,
EarlyStopping
from tensorflow.keras.optimizers import Adadelta, Adam, SGD
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D,
Dropout, Flatten, AveragePooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)
os.chdir("/content/gdrive/My Drive/Colab Notebooks") # might have to
change path to point to your Colab Notebooks folder
 Load and format data
data = np.load('/content/gdrive/MyDrive/Colab
Notebooks/MNIST_CorrNoise.npz')
x_train = data['x_train']
y_train = data['y_train']
num_cls = len(np.unique(y_train))
print('Number of classes: ' + str(num_cls))
print('Example of handwritten digit with correlated noise: \n')
k = 3000
plt.imshow(np.squeeze(x_train[k,:,:]))
plt.show()
print('Class: '+str(y_train[k])+'\n')
# RESHAPE and standarize
x_train = np.expand_dims(x_train/255,axis=3)
# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_cls)
print('Shape of x_train: '+str(x_train.shape))
print('Shape of y_train: '+str(y_train.shape))
 Output on running above code segment:
Number of classes: 10
Example of handwritten digit with correlated noise:
Class: 9
Shape of x_train: (60000, 28, 28, 1) Shape of y_train: (60000, 10)
 
 Training
def cnn_model():
 input_shape = x_train.shape[1:4] #(28,28,1)
 model = Sequential()
 #First convolutional layer
 model.add(Conv2D(32,
kernel_size=(3,3),padding='same',activation='linear',input_shape=input_sha
pe))
 model.add(LeakyReLU(alpha=0.1))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Dropout(0.25))
 #Second convolutional layer
 model.add(Conv2D(64, (3, 3),padding='same', activation='linear'))
 model.add(LeakyReLU(alpha=0.1))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Dropout(0.25))
 #Third convolutional layer
 model.add(Conv2D(128, (3, 3), padding='same',activation='linear'))
 model.add(LeakyReLU(alpha=0.1))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Dropout(0.4))
 model.add(Flatten()) # transforms matrix feature map to vector for dense
layer (fully connected)
 model.add(Dense(128, activation='linear'))
 model.add(LeakyReLU(alpha=0.1))
 model.add(Dropout(0.3))
 model.add(Dense(num_cls, activation='softmax'))
 model.summary()
return model
  
 model = Sequential()
model.compile(loss = categorical_crossentropy, optimizer = Adadelta(),
metrics= ['accuracy'])
model.compile(loss = categorical_crossentropy, optimizer = SGD(), metrics
= ['accuracy'])
model.compile(loss = categorical_crossentropy, optimizer = Adam(), metrics
= ['accuracy'])
def ModelCheckpointer():
 checkpointer = ModelCheckpoint(filepath = pweight, verbose=1,
save_best_only=True)
 callbacks_list = [checkpointer]
 history=model.fit(x_train, y_train, epochs=ep, batch_size=b_size,
verbose=1, shuffle=True, validation_split = val_split,
callbacks=callbacks_list)
 print('CNN weights saved in ' + pweight)
 # Plot loss vs epochs
 plt.plot(history.history['loss'])
 plt.plot(history.history['val_loss'])
 plt.title('model loss')
 plt.ylabel('loss')
 plt.xlabel('epoch')
 plt.legend(['train', 'val'], loc='upper right')
 plt.show()
 # Plot accuracy vs epochs
 plt.plot(history.history['accuracy'])
 plt.plot(history.history['val_accuracy'])
 plt.title('model accuracy')
 plt.ylabel('accuracy')
 plt.xlabel('epoch')
 plt.legend(['train', 'val'], loc='upper left')
 plt.show()
return
 
 Make predictions in test set
def testing():
 from keras.models import load_model
## LOAD DATA
 data = np.load('./MNIST_CorrNoise.npz')
 x_test = data['x_test']
 y_test = data['y_test']
 num_cls = len(np.unique(y_test))
 print('Number of classes: ' + str(num_cls))
 # RESHAPE and standarize
 x_test = np.expand_dims(x_test/255,axis=3)
 print('Shape of x_train: '+str(x_test.shape)+'\n')
 ## Define model parameters
 pweight='/content/gdrive/MyDrive/Colab Notebooks/weights/weights_' +
model_name + '.hdf5'
 model = load_model(pweight)
 y_p = model.predict(x_test)
 y_pred = np.argmax(y_p, axis=1)
 y_pred.shape
 y_test.shape
 Acc_pred = sum(y_pred == y_test)/len(y_test)
 print('Accuracy in test set is: ',Acc_pred)
 return
 Training Model
model_name='CNN' # To compare models, you can give them different names
pweight='/content/gdrive/MyDrive/Colab Notebooks/weights/weights_' +
model_name + '.hdf5'
if not os.path.exists('/content/gdrive/MyDrive/Colab Notebooks/weights'):
 
  os.mkdir('/content/gdrive/MyDrive/Colab Notebooks/weights')
## EXPLORE VALUES AND FIND A GOOD SET
b_size = 30 # batch size
val_split = 0.6 # percentage of samples used for validation (e.g. 0.5);
ep = 20 # number of epochs
model=cnn_model()
model.compile(loss=categorical_crossentropy,
   optimizer=Adam(),
   metrics=['accuracy'])
ModelCheckpointer()
# x=model.evaluate(x_train, y_train, verbose=0)
# print("test loss, test acc:",x )
testing()
 Output on running above code segment:
Model: "sequential_1" _________________________________________________________________
Layer (type) Output Shape Param # =================================================================
conv2d (Conv2D) (None, 28, 28, 32) 320 leaky_re_lu (LeakyReLU) (None, 28, 28, 32) 0
max_pooling2d (MaxPooling2D (None, 14, 14, 32) 0 )
dropout (Dropout) (None, 14, 14, 32)
conv2d_1 (Conv2D) (None, 14, 14, 64)
leaky_re_lu_1 (LeakyReLU) (None, 14, 14, 64)
0 18496
0
max_pooling2d_1 (MaxPooling (None, 7, 7, 64)
0
2D)
dropout_1 (Dropout) (None, 7, 7, 64) conv2d_2 (Conv2D) (None, 7, 7, 128) leaky_re_lu_2 (LeakyReLU) (None, 7, 7, 128)
0 73856
0

 max_pooling2d_2 (MaxPooling (None, 3, 3, 128) 0 2D)
dropout_2 (Dropout)
flatten (Flatten)
dense (Dense)
leaky_re_lu_3 (LeakyReLU) (None, 128) dropout_3 (Dropout) (None, 128) dense_1 (Dense) (None, 10)
0 0
147584 0
0 1290
(None, 3, 3, 128) (None, 1152)
(None, 128)
================================================================= Total params: 241,546
Trainable params: 241,546
Non-trainable params: 0 _________________________________________________________________
Epoch 1/20
794/800 [============================>.] - ETA: 0s - loss: 1.6789 - accuracy: 0.3978
Epoch 00001: val_loss improved from inf to 0.76001, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 23s 17ms/step - loss: 1.6734 - accuracy: 0.3996 - val_loss: 0.7600 - val_accuracy: 0.7554
Epoch 2/20
795/800 [============================>.] - ETA: 0s - loss: 0.8100 - accuracy: 0.7268
Epoch 00002: val_loss improved from 0.76001 to 0.53879, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 11s 14ms/step - loss: 0.8093 - accuracy: 0.7270 - val_loss: 0.5388 - val_accuracy: 0.8194
Epoch 3/20
800/800 [==============================] - ETA: 0s - loss: 0.6474 - accuracy: 0.7821
Epoch 00003: val_loss improved from 0.53879 to 0.45768, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 14s 18ms/step - loss: 0.6474 - accuracy: 0.7821 - val_loss: 0.4577 - val_accuracy: 0.8461
Epoch 4/20
797/800 [============================>.] - ETA: 0s - loss: 0.5587 - accuracy: 0.8121
Epoch 00004: val_loss improved from 0.45768 to 0.42932, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 13s 16ms/step - loss: 0.5590 - accuracy: 0.8120 - val_loss: 0.4293 - val_accuracy: 0.8563

 Epoch 5/20
799/800 [============================>.] - ETA: 0s - loss: 0.5067 - accuracy: 0.8282 Epoch 00005: val_loss improved from 0.42932 to 0.37829, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 13s 16ms/step - loss: 0.5067 - accuracy: val_loss: 0.3783 - val_accuracy: 0.8733
Epoch 6/20
799/800 [============================>.] - ETA: 0s - loss: 0.4743 - accuracy: 0.8415 Epoch 00006: val_loss improved from 0.37829 to 0.37072, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 13s 16ms/step - loss: 0.4742 - accuracy: val_loss: 0.3707 - val_accuracy: 0.8772
Epoch 7/20
797/800 [============================>.] - ETA: 0s - loss: 0.4450 - accuracy: 0.8481 Epoch 00007: val_loss improved from 0.37072 to 0.35215, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 13s 16ms/step - loss: 0.4449 - accuracy: val_loss: 0.3522 - val_accuracy: 0.8828
Epoch 8/20
797/800 [============================>.] - ETA: 0s - loss: 0.4242 - accuracy: 0.8547 Epoch 00008: val_loss improved from 0.35215 to 0.34359, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 13s 16ms/step - loss: 0.4239 - accuracy: val_loss: 0.3436 - val_accuracy: 0.8853
Epoch 9/20
798/800 [============================>.] - ETA: 0s - loss: 0.4059 - accuracy: 0.8622 Epoch 00009: val_loss improved from 0.34359 to 0.33216, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 11s 14ms/step - loss: 0.4061 - accuracy: val_loss: 0.3322 - val_accuracy: 0.8874
Epoch 10/20
795/800 [============================>.] - ETA: 0s - loss: 0.3836 - accuracy: 0.8691 Epoch 00010: val_loss improved from 0.33216 to 0.33130, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 13s 16ms/step - loss: 0.3843 - accuracy: val_loss: 0.3313 - val_accuracy: 0.8896
Epoch 11/20
800/800 [==============================] - ETA: 0s - loss: 0.3740 - accuracy: 0.8715 Epoch 00011: val_loss improved from 0.33130 to 0.32313, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 13s 16ms/step - loss: 0.3740 - accuracy: val_loss: 0.3231 - val_accuracy: 0.8929
Epoch 12/20
795/800 [============================>.] - ETA: 0s - loss: 0.3540 - accuracy: 0.8797
0.8282 -
0.8414 -
0.8480 -
0.8549 -
0.8620 -
0.8687 -
0.8715 -

 Epoch 00012: val_loss improved from 0.32313 to 0.31229, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 12s 16ms/step - loss: 0.3551 - accuracy: val_loss: 0.3123 - val_accuracy: 0.8963
Epoch 13/20
795/800 [============================>.] - ETA: 0s - loss: 0.3496 - accuracy: 0.8790 Epoch 00013: val_loss did not improve from 0.31229
800/800 [==============================] - 12s 15ms/step - loss: 0.3496 - accuracy: val_loss: 0.3176 - val_accuracy: 0.8935
Epoch 14/20
795/800 [============================>.] - ETA: 0s - loss: 0.3496 - accuracy: 0.8830 Epoch 00014: val_loss did not improve from 0.31229
800/800 [==============================] - 11s 13ms/step - loss: 0.3491 - accuracy: val_loss: 0.3201 - val_accuracy: 0.8917
Epoch 15/20
796/800 [============================>.] - ETA: 0s - loss: 0.3375 - accuracy: 0.8827 Epoch 00015: val_loss did not improve from 0.31229
800/800 [==============================] - 12s 15ms/step - loss: 0.3381 - accuracy: val_loss: 0.3150 - val_accuracy: 0.8966
Epoch 16/20
796/800 [============================>.] - ETA: 0s - loss: 0.3246 - accuracy: 0.8874 Epoch 00016: val_loss improved from 0.31229 to 0.30988, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 12s 16ms/step - loss: 0.3247 - accuracy: val_loss: 0.3099 - val_accuracy: 0.8974
Epoch 17/20
795/800 [============================>.] - ETA: 0s - loss: 0.3241 - accuracy: 0.8883 Epoch 00017: val_loss did not improve from 0.30988
800/800 [==============================] - 12s 15ms/step - loss: 0.3239 - accuracy: val_loss: 0.3241 - val_accuracy: 0.8967
Epoch 18/20
798/800 [============================>.] - ETA: 0s - loss: 0.3151 - accuracy: 0.8901 Epoch 00018: val_loss did not improve from 0.30988
800/800 [==============================] - 12s 15ms/step - loss: 0.3151 - accuracy: val_loss: 0.3112 - val_accuracy: 0.8977
Epoch 19/20
800/800 [==============================] - ETA: 0s - loss: 0.3055 - accuracy: 0.8949 Epoch 00019: val_loss improved from 0.30988 to 0.30944, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 11s 14ms/step - loss: 0.3055 - accuracy: val_loss: 0.3094 - val_accuracy: 0.8992
Epoch 20/20
796/800 [============================>.] - ETA: 0s - loss: 0.3078 - accuracy: 0.8932
0.8797 -
0.8790 -
0.8831 -
0.8826 -
0.8874 -
0.8884 -
0.8902 -
0.8949 -

 Epoch 00020: val_loss improved from 0.30944 to 0.30695, saving model to /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
800/800 [==============================] - 11s 14ms/step - loss: 0.3077 - accuracy: 0.8932 - val_loss: 0.3069 - val_accuracy: 0.9001
CNN weights saved in /content/gdrive/MyDrive/Colab Notebooks/weights/weights_CNN.hdf5
Number of classes: 10
Shape of x_train: (10000, 28, 28, 1)
Accuracy in test set is: 0.9044
   
 from keras.layers import Activation
data = np.load('./MNIST_CorrNoise.npz')
x_test = data['x_test']
y_test = data['y_test']
x_test = np.expand_dims(x_test/255,axis=3)
x=x_test[10]
plt.subplot(1, 4, 1)
#print(y_test[10])
plt.imshow(x[:,:,0])
plt.axis("off")
plt.gca().set_title('Original = 0')
x.astype('float32')
model = Sequential()
model.add(Conv2D(1,kernel_size=(3, 3),input_shape=x.shape))
x_batch=np.expand_dims(x,axis=0)
g=model.predict(x_batch)
conv_x=model.predict(x_batch)
conv_x = np.squeeze(conv_x, axis=0)
conv_x = np.squeeze(conv_x, axis=2)
plt.subplot(1,4,2)
plt.imshow(conv_x)
plt.axis("off")
plt.gca().set_title('Conv2D')
model = Sequential()
model.add(Conv2D(1,kernel_size=(3, 3),input_shape=x.shape))
model.add(Conv2DTranspose(filters=1, kernel_size=(3,3), strides=(1,1),
padding='same'))
x_batch=np.expand_dims(x,axis=0)
outputs=model.predict(x_batch)
outputs = np.squeeze(outputs, axis=0)
outputs = np.squeeze(outputs, axis=2)
plt.subplot(1, 4, 3)
plt.imshow(outputs)
plt.axis("off")
plt.gca().set_title('Conv2DTranspose')
model = Sequential()
model.add(Conv2D(1,kernel_size=(3, 3),input_shape=x.shape))
model.add(Activation('relu'))
x_batch=np.expand_dims(x,axis=0)
g=model.predict(x_batch)
 
conv_x=model.predict(x_batch)
conv_x = np.squeeze(conv_x, axis=0)
conv_x = np.squeeze(conv_x, axis=2)
plt.subplot(1, 4, 4)
plt.imshow(conv_x)
plt.axis("off")
plt.gca().set_title('relu')
plt.tight_layout()
 
