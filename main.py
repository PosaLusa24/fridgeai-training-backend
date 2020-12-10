import json
import os
import time
from zipfile import ZipFile
from MiniVGG_train import train
import tensorflow as tf
from datetime import datetime

SERVED_PATH = 'served'
ITEMS_PATH = 'items'


print('>> Welcome to FridgeAI Training Server!')
print('>> Using served folder:', SERVED_PATH)
print('Waiting for data from client...')
while True:
    zipfiles = [
        file for file in os.listdir(SERVED_PATH)
        if os.path.splitext(file)[1] == '.zip'
    ]
    if zipfiles:
        print('>> Zipfile received. Extracting...')
        zipfile_path = os.path.join(SERVED_PATH, zipfiles[0])
        category_path = os.path.join(ITEMS_PATH, zipfiles[0][:-3])
        with ZipFile(zipfile_path, 'r') as zip:
            zip.extractall(category_path)
        os.remove(zipfile_path)
        print('>> Successfully extracted data. Commencing training...')

        tflite_path = os.path.join(
            SERVED_PATH,
            datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '.tflite'
        )

        tflites = [
            file for file in os.listdir(SERVED_PATH)
            if os.path.splitext(file)[1] == '.tflite'
        ]
        for file in tflites:
            os.remove(os.path.join(SERVED_PATH, file))

        # Comment these lines and uncomment below code to test with
        # generated dummy model
        train(ITEMS_PATH, SERVED_PATH)
        converter = tf.lite.TFLiteConverter.from_saved_model(SERVED_PATH)
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as file:
            file.write(tflite_model)
        with open(os.path.join(SERVED_PATH, 'labels.json'), 'w') as file:
            json.dump(os.listdir(ITEMS_PATH), file)
        # with open(tflite_path, 'w') as file:
        #     file.write('test')
        # with open(os.path.join(SERVED_PATH, 'labels.json'), 'w') as file:
        #     json.dump(os.listdir(ITEMS_PATH), file)

        print('>> Training complete. Model updated. Waiting for more data...')
    time.sleep(1)
