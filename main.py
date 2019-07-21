import keras
import tensorflow as tf
from keras.layers import Conv2D, Lambda, Dense, Flatten
from keras.layers import Activation, MaxPooling2D, Input, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LearningRateScheduler,ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob,os
from keras.regularizers import l2
from math import ceil
from keras.datasets import cifar10, mnist
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.backend.tensorflow_backend import set_session
from keras.applications.resnet50 import ResNet50
import argparse

parser = argparse.ArgumentParser(description = "resume")
parser.add_argument('--resume', action = 'store_true', default = False, help = 'if resume')
parser.add_argument('--model-path', type=str, default = "./checkpoint")
args = parser.parse_args()

batch_size = 8
epochs = 150
data_augmentation = True
num_classes = 6

# 使用第二块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


train_datagen = ImageDataGenerator( rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    featurewise_center=False,
                                    samplewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    zca_epsilon=1e-06,
                                    rotation_range=0,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    channel_shift_range=0.,
                                    fill_mode='nearest',
                                    cval=0.,
                                    vertical_flip=False,
                                    preprocessing_function=None,
                                    data_format=None)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('./data/train',
                                                    target_size=(500,500),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('./data/validation',
                                                        target_size=(500,500),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

input_shape = (500,500,3)

# for i in range(9):
#     train_generator.next()

# # 找到本地生成图，把9张图打印到同一张figure上
# name_list = glob.glob('./data/train/'+'1/*')
# fig = plt.figure()
# for i in range(9):
#     img = Image.open(name_list[i])
#     sub_img = fig.add_subplot(331 + i)
#     sub_img.imshow(img)
# plt.show()




def lr_schedule(epoch):
    lr = 1e-4
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

#model = resnet_v1(input_shape=input_shape, depth=20 ,num_classes=6)

model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None, input_tensor=None, input_shape=input_shape, pooling=None, classes=6)

if args.resume:
    tmp_path = args.model_path
    model = load_model(tmp_path)

model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer="adam",
                metrics=['accuracy'])

model.summary()

save_dir = os.path.join(os.getcwd(), 'checkpoint')
model_name = 'InceptionResNetV2_{epoch:03d}.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)

cpt = ModelCheckpoint(filepath = filepath, 
                            monitor ='val_acc',
                            verbose = 1,
                            save_best_only = False)

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(monitor = 'val_loss',
                               factor = np.sqrt(0.1),
                               cooldown=0,
                               patience = 5,
                               min_lr = 0.5e-8)

# eystp = keras.callbacks.EarlyStopping(monitor='val_loss', 
#                                 min_delta=0, patience=0, 
#                                 verbose=0, mode='auto') 
#                                 #baseline=None, 
#                                 #restore_best_weights=False)


tb = TensorBoard(log_dir='./logs', histogram_freq=0,
                batch_size=batch_size, 
                write_graph=True, write_grads=False, 
                write_images=False, embeddings_freq=0, 
                embeddings_layer_names=None, 
                embeddings_metadata=None) 

cbs = [cpt, lr_reducer, lr_scheduler, tb]

data_aug = True

if not data_aug: 
    # history = model.fit(train_images,
    #         train_labels,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         verbose=1,
    #         validation_data=(test_images,test_labels),
    #         shuffle=True
    #         )
    pass
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
        data_format=None)
    
    #datagen.fit(train_images)

    history = model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        epochs = epochs, 
                        verbose = 1, 
                        workers = 1, 
                        callbacks = cbs,
                        steps_per_epoch = 30000,
                        samples_per_epoch=2000,
                        validation_steps = 100)

#score = model.evaluate(validation_generator,verbose=1)

# print('test loss:',score[0])
# print('test accuracy:',score[1])

import matplotlib.pyplot as plt

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("weights.h5")


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
plot_model(model, to_file='model.png')
