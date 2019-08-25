#  [ { "image_id": "prcv2019test05213.jpg", "disease_class":1 }, ...]

import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import *
from keras.applications.imagenet_utils import decode_predictions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
testPath = "./data/test"
modelName = "/home/dongpeijie/classify.keras/checkpoint/dpj_005.h5"

# Loading model from h5......
print("Loading model from h5......")
model = load_model(modelName)

# Loading image ......
img = image.load_img("./data/test/DSC16_02434.JPG", target_size=(500, 500))

# turn into 4d tensor形式
x = image.img_to_array(img)
print("Before x:shape ",x.shape)
x = np.expand_dims(x, axis=0)
print("After x:shape ",x.shape)

preds = model.predict(x)
print(np.argmax(preds))
print(preds)
#print('Predicted:', decode_predictions(y, top=3)[0])

'''
model = load_model(modelName)

imgData = cv2.imread("./data/test/DSC16_02434.JPG",1)

arr = np.asarray(imgData, dtype = "float32")

testData[0,:,:,:] = arr

#testData = testData.reshape(testData.shape[0], 500, 500, 3)

out = model.predict_classes(testData, batch_size=1, verbose=1)
print(out)

'''
'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(modelName)
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
'''
