import keras
import tensorflow as tf
from keras.layers import Conv2D, Lambda, Dense, Flatten
from keras.layers import Activation, MaxPooling2D, Input
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
from keras.regularizers import l2
import os
from keras.datasets import cifar10, mnist
from keras.utils import plot_model

batch_size = 20
epochs = 10
data_augmentation = True
num_classes = 10

(train_images, train_labels) , (test_images,test_labels)=cifar10.load_data()

img_row,img_col,channel = 64,64,1

input_shape = train_images.shape[1:]

# input_shape = (img_row,img_col,1)

#将数据维度进行处理
# train_images = train_images.reshape(train_images.shape[0],img_row,img_col,channel)
# test_images = test_images.reshape(test_images.shape[0],img_row,img_col,channel)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

## 进行归一化处理
train_images  /= 255
test_images /= 255

# 将类向量，转化为类矩阵
# 从 5 转换为 0 0 0 0 1 0 0 0 0 0 矩阵
train_labels = keras.utils.to_categorical(train_labels,num_classes)
test_labels = keras.utils.to_categorical(test_labels,num_classes)


"""
构造网络结构
"""
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes =10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    num_filters = 16
    num_res_blocks = int((depth-2)/6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs = x,num_filters = num_filters, strides = strides)
            y = resnet_layer(inputs = y, num_filters = num_filters, activation =None)
            
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, 
                                num_filters = num_filters, 
                                kernel_size = 1,
                                strides = strides,
                                activation = None,
                                batch_normalization = True)
            x = keras.layers.add([x,y])
            x = keras.layers.Activation('relu')(x)
        num_filters *= 2

    x = keras.layers.AveragePooling2D(pool_size=7)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax',
                    kernel_initializer = 'he_normal')(y)
    model = Model(inputs=inputs, outputs = outputs)
    return model

model = resnet_v1(input_shape=input_shape, depth=20)

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=Adam(lr = lr_schedule(0)),
                metrics=['accuracy'])

model.summary()

save_dir = os.path.join(os.getcwd(), 'checkpoint')
model_name = 'cifar_{epoch:03d}.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath = filepath, monitor ='val_acc',
                            verbose = 1,
                            save_best_only = True)

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1),
                               cooldown=0,
                               patience = 5,
                               min_lr = 0.5e-6)

tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, 
                write_graph=True, write_grads=False, 
                write_images=False, embeddings_freq=0, 
                embeddings_layer_names=None, 
                embeddings_metadata=None, 
                embeddings_data=None, update_freq='epoch')

cbs = [checkpoint, lr_reducer, lr_scheduler, tb]

data_aug = True

if not data_aug: 
    history = model.fit(train_images,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(test_images,test_labels),
            shuffle=True
            )
else:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images  
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    
    datagen.fit(train_images)

    history = model.fit_generator(datagen.flow(train_images[1:10000], train_labels[1:10000], batch_size = batch_size),
                        validation_data=(test_images[1:1000], test_labels[1:1000]),
                        epochs=epochs, 
                        verbose=1, 
                        workers = 1, 
                        callbacks = cbs,
                        steps_per_epoch=10000)

score = model.evaluate(test_images,test_labels,verbose=1)

print('test loss:',score[0])
print('test accuracy:',score[1])

import matplotlib.pyplot as plt


plot_model(model, to_file='model.png')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("trainTestAcc.png")
#plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("trainTestLoss.png")
#plt.show()