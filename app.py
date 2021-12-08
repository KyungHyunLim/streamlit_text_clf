import streamlit as st
import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.title("악성 댓글 분류 하기")

@st.cache
def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model('beomi/KcELECTRA-base')

text_input = st.text_input("텍스트를 입력해주세요")
st.write(text_input)