import cv2, numpy as np
import base64
from PIL import Image
from io import BytesIO
from PIL import Image, ImageCms, ImageOps
import replicate
import os
from loguru import logger
import requests
import shlex, subprocess, argparse
import time

start = time.time()

os.environ["REPLICATE_API_TOKEN"] = "c1bb1da81e021a022c7cff6e6dc03d320a4bb2db"

model = replicate.models.get("xinntao/gfpgan")
version = model.versions.get("6129309904ce4debfde78de5c209bce0022af40e197e132f08be8ccce3050393")

parser = argparse.ArgumentParser(description='Parse path')
parser.add_argument('--path', type=str)

args = parser.parse_args()

impath = args.path


'''
Super Resolution and Face Restore using GFP Gan (Replicate API)
'''
uid = impath.split('/')[-1].split('.')[0]
output = version.predict(img=open(impath, "rb"), scale=4)
logger.info('Replicate Image request sent')
if 'selfie' in impath:
    filename = f'sr_selfie_images/{uid}.jpg'
else:
    filename = f'sr_id_images/{uid}.jpg'
response = requests.get(output)
#im = response.content.read()
temp_img = np.array(Image.open(BytesIO(response.content)))
temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
cv2.imwrite(filename, temp_img)
logger.info('SR Image saved')
end = time.time()
logger.info(f"total time: {end - start}")