import streamlit as st
import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer

dicts = {
    0: 'None',
    1: 'Offensive',
    2: 'Attack'
}

st.title("악성 댓글 분류 하기")

@st.cache
def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

model = load_model('beomi/KcELECTRA-base')
tokenizer = load_tokenizer('beomi/KcELECTRA-base')

text_input = st.text_input("텍스트를 입력해주세요")
if st.button("분류해보기!"):
    inputs = tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50,
            add_special_tokens=True,
        )
    pred = model(**inputs)
    classes = np.argmax(pred['logits'].detach().numpy())
    st.write(dicts[classes])