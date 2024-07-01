import os
import cv2
import json
import numpy as np
from tqdm import tqdm

result_file = '/home/tungdop2/BLIP/output/UIT_ViIC/result/val_epoch19.json'
root = '/home/tungdop2/BLIP/source/UIT-ViIC/images'
target = '/home/tungdop2/BLIP/output/UIT_ViIC/demo/val'
os.makedirs(target, exist_ok=True)

import unidecode
def remove_accent(text):
    return unidecode.unidecode(text)

with open(result_file, 'r') as f:
    results = json.load(f)

for result in tqdm(results):
    image_id = result['image_id']
    caption = result['caption'].lower()
    caption = remove_accent(caption)
    # print(caption)

    imgae_path = '0' *(12 - len(str(image_id))) + str(image_id) + '.jpg'
    imgae_path = os.path.join(root, imgae_path)
    # print(imgae_path)

    # show image in 5s
    img = cv2.imread(imgae_path)
    img = cv2.resize(img, (640, 480))
    caption_background = np.ones((50, 640, 3), dtype=np.uint8) * 255
    cv2.putText(caption_background, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    img = np.concatenate((img, caption_background), axis=0)
    # save image to target folder
    cv2.imwrite(os.path.join(target, imgae_path.split('/')[-1]), img)
