# pip install fastapi uvicorn torch transofmrers PyPDF2 aiofiles scikit-leanr nltk python-multipart 
from fastapi import FastAPI , File , UploadFile , Form 
from fastapi.responses import HTMLResponse , JSONResponse
from fastapi.templating import Jinja2Templates
from transformers import BertForSequenceClassification , BertTokenizer 
import torch
import PyPDF2
from io import BytesIO
from fastapi import Request
import pickle 
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

#load model and tokenizer
model = BertForSequenceClassification.from_pretrained("patient_model")
tokenizer = BertTokenizer.from_pretrained('patient_model')
label_encoder = pickle.load(open("label_encoder.pkl" , 'rb'))




app = FastAPI()

templates = Jinja2Templates(directory='templates')

#end points  rools urls

@app.get('/', response_class=HTMLResponse )
async def upload_form(request:Request):
    return templates.TemplateResponse('index.html' , {"request": request})