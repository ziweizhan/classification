# -*- coding:utf-8 -*-
# @time :2021.3.15
# @IDE : pycharm
# @author :Ziwei.zhan

import requests
import time

REST_API_URL = 'http://127.0.0.1:5000/predict'

def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    r = requests.post(REST_API_URL, files=payload).json()
    if r['success']:
        for (i, result) in enumerate(r['predictions']):
            print(result)
    else:
        print('Request failed')


if __name__ == '__main__':
    t1 = time.time()
    img_path = 'dog.png'
    predict_result(img_path)
    t2 = time.time()
    print(t2-t1)
