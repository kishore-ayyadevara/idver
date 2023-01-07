from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import cv2, numpy as np
import base64
from PIL import Image
from io import BytesIO
from PIL import Image, ImageCms, ImageOps
import replicate
import os
from loguru import logger
import requests
import shlex, subprocess, time
# import get_sr
from utils import compare_two_images, check_template, check_fake

os.environ["REPLICATE_API_TOKEN"] = "c1bb1da81e021a022c7cff6e6dc03d320a4bb2db"

app = FastAPI()

model = replicate.models.get("xinntao/gfpgan")
version = model.versions.get("6129309904ce4debfde78de5c209bce0022af40e197e132f08be8ccce3050393")


        
@app.post('/selfie_image')
async def selfie_file_upload(selfie_image: UploadFile = File(...), uid: str = Form(...)):

    img = await selfie_image.read()
    temp_img = Image.open(BytesIO(img))
    profile_none_selfie = 0
    try:
        icc_selfie = temp_img.info.get('icc_profile')
        f_selfie = BytesIO(icc_selfie)
        prf_selfie = ImageCms.ImageCmsProfile(f_selfie)
        if prf_selfie.profile.model == None:
            profile_none_selfie = 0
        else:
            profile_none_selfie = 1
    except:
        pass

    selfie = Image.open(BytesIO(img))
    try:
        if profile_none_selfie == 0:
            selfie = ImageOps.exif_transpose(selfie)
    except:
        selfie = selfie.transpose(Image.ROTATE_270)

    selfie = np.array(selfie)

    if(profile_none_selfie == 0):
        selfie = cv2.cvtColor(selfie, cv2.COLOR_BGR2RGB)

    fpath = f"selfie_images/{uid}.jpg"
    
    cv2.imwrite(fpath, selfie)
    
    generate_number_script = f"python3 get_sr.py --path {fpath}"
    cmd = shlex.split(generate_number_script)
    subprocess.Popen(cmd, start_new_session=True)
    
    
    return {"status": 'DONE'}

@app.post('/id_image')
async def id_file_upload(id_image: UploadFile = File(...), uid: str = Form(...)):

    img = await id_image.read()
    temp_img = Image.open(BytesIO(img))
    profile_none_selfie = 0
    try:
        icc_selfie = temp_img.info.get('icc_profile')
        f_selfie = BytesIO(icc_selfie)
        prf_selfie = ImageCms.ImageCmsProfile(f_selfie)
        if prf_selfie.profile.model == None:
            profile_none_selfie = 0
        else:
            profile_none_selfie = 1
    except:
        pass

    id_image = Image.open(BytesIO(img))
    try:
        if profile_none_selfie == 0:
            id_image = ImageOps.exif_transpose(id_image)
    except:
        id_image = id_image.transpose(Image.ROTATE_270)

    id_image = np.array(id_image)

    if(profile_none_selfie == 0):
        id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)

    fpath = f"id_images/{uid}.jpg"
    
    cv2.imwrite(fpath, id_image)
    
    generate_number_script = f"python3 get_sr.py --path {fpath}"
    cmd = shlex.split(generate_number_script)
    subprocess.Popen(cmd, start_new_session=True)
    
    return {"status": 'DONE'}


@app.post('/verify')
async def verify(uid: str = Form(...), id_type: str = Form(...)):
    
    selfie_path = f"selfie_images/{uid}.jpg"
    id_path = f"id_images/{uid}.jpg"
    start = time.time()
    path_exists = ((os.path.exists('sr_'+selfie_path)) & (os.path.exists('sr_'+id_path)))
    logger.info(f"in {uid} and path exists is {path_exists}")
    while (path_exists == False):
        path_exists = ((os.path.exists('sr_'+selfie_path)) & (os.path.exists('sr_'+id_path)))
        logger.info(f"in loop {uid}")
        time.sleep(1)

    sr_selfie_path = 'sr_'+selfie_path
    sr_id_path = 'sr_'+id_path
    image_comparison = compare_two_images(sr_id_path, sr_selfie_path)
    end = time.time()
    logger.info(f"overall time taken: {end - start}")
    
    
    template_comparison = check_template(id_path, id_type)
    
    fake_detection = check_fake(sr_selfie_path)
        
        
    return {"status": 'DONE', 'face_verified': image_comparison['verified'], 
            'template_verified': template_comparison, 'fake': fake_detection}