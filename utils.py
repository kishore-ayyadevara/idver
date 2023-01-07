from deepface import DeepFace
from PIL import Image
from fuzzywuzzy import fuzz
from datetime import datetime
# import PaddleOCR
from loguru import logger
import cv2, torch, numpy as np, os
from test import test

id2label = {"aadhar": 2, "pan": 3}
model = torch.load("models/id_identifier.pth",
                       map_location=torch.device('cpu'))

def preprocess_image(img):
    img = torch.tensor(img).permute(2, 0, 1)
    return img.to("cpu").float()


def compare_two_images(img_path1, img_path2):
    #models = ["ArcFace", "VGG-Face", "Facenet512", "Facenet", "SFace"]
    models = ["ArcFace"]
    result = {}
    true_count = 0
    false_count = 0
    for model in models:
        try:
            result = DeepFace.verify(img1_path=img_path1, img2_path=img_path2, model_name=model,
                                     distance_metric="euclidean_l2", detector_backend='opencv')
        except ValueError:
            pass
        if result:
            if result["verified"]:
                true_count += 1
            else:
                false_count += 1

        logger.info(f'{model} : {result}')
    logger.info(f'True Count : {true_count}, False Count : {false_count}')
    if not result:
        return {"verified": "Please upload a better quality picture!"}

    if true_count >= 1:
        return {"verified": True}
    else:
        return {"verified": False}
    
def check_template(img_path, id_type):    
    img = Image.open(img_path).convert("RGB")
    img = np.array(img.resize((224, 224), resample=Image.BILINEAR)) / 255.

    img = preprocess_image(img)
    outputs = model([img])

    # np_boxes = outputs[0]['boxes'].detach().cpu().numpy()
    np_labels = outputs[0]['labels'].detach().cpu().numpy()
    np_scores = outputs[0]['scores'].detach().cpu().numpy()
    score = np_scores[np_labels == id2label[id_type]]
    print(np_labels, np_scores)
    if (score > 0.7):
        result = True
    else:
        result = False
    return result

def check_fake(img_path):
#     cmd = f'python test.py --image_name {img_path}'
#     fake = os.system(cmd)
    fake = test(img_path)
    print('fake check:', fake)
    return fake