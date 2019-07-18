import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import *
from keras.applications.imagenet_utils import decode_predictions
import json

file_list = os.listdir('data/test/')
images = []
 
modelName = "/home/dongpeijie/classify.keras/checkpoint/cifar_005.h5"

# Loading model from h5......
print("Loading model from h5......")
model = load_model(modelName)

for file in file_list:
    # print(file)
    img = image.load_img(os.path.join('data/test/', file), target_size=(500, 500))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
 
x_train = np.array(images, dtype="float") / 255.0
x = np.concatenate([x for x in x_train])
 
#预测
y = model.predict(x)
 
 
#根据结果可以看出来，0代表的是猫，1代表的是狗。
# 同时也可以从训练cats_and_dogs_small/train/里面文件的顺序知道类别代表的信息
print(y)

f = open("./submit.json", "w")


for i in range(len(file_list)):
    #print(y[i][0])
    print('image: {} \t class: {}'.format(file_list[i],np.argmax(y[i])))
    # [ { "image_id": "prcv2019test05213.jpg", "disease_class":1 }, ...]

    new_dict = {"image_id" : file_list[i], "disease_class":int(np.argmax(y[i]))}
    json.dump(new_dict,f)
    #print('image class:', int(y[i]))
    #print('image class:', round(y[i]))
    # if y[i][0] > 0.5:
    #     print('image {} class:'.format(file_list[i]), 1)
    # else:
    #     print('image {} class:'.format(file_list[i]), 0)

f.close()