import os
import cv2
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import *
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf

valid_path = "/home/dongpeijie/classify.keras/data/train"
modelName = "./checkpoint/ResNet50_108.h5"#"./checkpoint/old/InceptionV3/InceptionV3_100.h5"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


# Loading model from h5......
print("Loading model from h5......")
model = load_model(modelName)

cnt_true = 0
cnt_false = 0

false_list = []

for i in os.listdir(valid_path):
    pa = os.path.join(valid_path, i) # folder
    for j in os.listdir(pa):
        pb = os.path.join(pa, j) # file
        img = image.load_img(pb, target_size=(500, 500))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        x = np.array(img,dtype="float") / 255.0
        y = model.predict(x)
        print(str(np.argmax(y[0])+1) == i,'\t',np.argmax(y[0])+1,'\t',pb)
        if str(np.argmax(y[0])+1) == i:
            cnt_true += 1
        else:
            cnt_false += 1
            false_list.append(pb)
print("==== results ====")
print(" True: {} \n False: {} \n Precision: {}".format(cnt_true,cnt_false,float(cnt_true)/float(cnt_true+cnt_false)))
print("==== results ====\n")

with open("./errorlist.txt", 'w') as f:
    for item in false_list:
        #print(item)
        f.write(item+"\n")
