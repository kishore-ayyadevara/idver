from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import cv2, numpy as np
import base64
from PIL import Image
from io import BytesIO

app = FastAPI()

        
@app.post('/file')
async def _file_upload(my_file: UploadFile = File(...), uid: str = Form(...)):
    contents = await my_file.read()
    nparr = np.array(Image.open(BytesIO(contents)))
    #img = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB)
    img = nparr
    print(img.shape, uid)
    
    _, encoded_img = cv2.imencode('.PNG', img)

    encoded_img = base64.b64encode(encoded_img)

    return{
        'filename': my_file.filename,
        'encoded_img': encoded_img
    }
