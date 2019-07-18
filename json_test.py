#  [ { "image_id": "prcv2019test05213.jpg", "disease_class":1 }, ...]

import os

testPath = ".\\data\\test"

for i in os.listdir(testPath):
    print(i)