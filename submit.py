import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import *
from keras.applications.imagenet_utils import decode_predictions
import json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

file_list = os.listdir('data/test/')
images = []
 
model1Name = "/home/dongpeijie/classify.keras/checkpoint/old/dpj/dpj_005.h5"
model2Name = "/home/dongpeijie/classify.keras/checkpoint/resnet50/ResNet50_108.h5"
model3Name = "/home/dongpeijie/classify.keras/checkpoint/inceptionv3/Inceptionv3_106.h5"
model4Name = "/home/dongpeijie/classify.keras/checkpoint/inceptionresnetv2/InceptionResNetv2_113.h5"
model5Name = "/home/dongpeijie/classify.keras/checkpoint/xception/Xception_105.h5"

# Loading model from h5......
print("Loading model from h5......")
model1 = load_model(model1Name)
model2 = load_model(model2Name)
model3 = load_model(model3Name)
model4 = load_model(model4Name)
model5 = load_model(model5Name)


for file in file_list:
    img = image.load_img(os.path.join('data/test/', file), target_size=(500, 500))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
 
x_train = np.array(images, dtype="float") / 255.0
x = np.concatenate([x for x in x_train])
 
#棰勬祴
y1 = model1.predict(x)
y2 = model2.predict(x)
y3 = model3.predict(x)
y4 = model4.predict(x)
y5 = model5.predict(x)

# print(y1,y2,y3)

f = open("./submit.json", "w")



for i in range(len(file_list)):
    #print(y[i][0])
    ton = [0,0,0,0,0,0]
    print('image: {} \t class: {}:{}:{}:{}:{}'.format(file_list[i],np.argmax(y1[i]),np.argmax(y2[i]),np.argmax(y3[i]),np.argmax(y4[i]),np.argmax(y5[i])))
    # [ { "image_id": "prcv2019test05213.jpg", "disease_class":1 }, ...]
    
    ton[np.argmax(y1[i])] += 1
    ton[np.argmax(y2[i])] += 1
    ton[np.argmax(y3[i])] += 1
    ton[np.argmax(y4[i])] += 1
    ton[np.argmax(y5[i])] += 1

    new_dict = {"image_id" : file_list[i], "disease_class":int(np.argmax(ton)+1)}
    json.dump(new_dict,f)

del model1
del model2
del model3
del model4
del model5

f.close()
