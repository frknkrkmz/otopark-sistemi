import streamlit as st
import cv2
import re
import numpy as np
from paddleocr import PaddleOCR
import pandas as pd
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Plaka Tanıma Sistemi", layout="wide")

st.title("🚗 Toplu Plaka Tanıma Sistemi")

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

ocr = load_ocr()

# Plaka Kontrol Fonksiyonu
def plaka_kontrol(metin):
    temiz_metin = re.sub(r'[^A-Z0-9]', '', metin.upper())
    model = r"^\d{2}[A-Z]{1,3}\d{2,5}$"
    if re.match(model, temiz_metin):
        return True, temiz_metin
    return False, None

uploaded_files = st.file_uploader(
    "Fotoğrafları seçin",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:

    veriler = []
    progress_bar = st.progress(0)
    toplam = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file)
        image_np = np.array(image)

        bulunan_plaka = "TESPIT_EDILEMEDI"
        en_yuksek_skor = 0.0

        result = ocr.ocr(image_np, cls=True)

        if result and result[0]:
            for line in result[0]:
                metin = line[1][0]
                skor = line[1][1]

                yasakli_kelimeler = ["OTOPARK", "TURKIYE", "TR", "ISTANBUL", "BURSA", "CADDE", "SOKAK"]
                if any(y in metin.upper() for y in yasakli_kelimeler):
                    continue

                uygun_mu, temiz_plaka = plaka_kontrol(metin)

                if uygun_mu and skor > en_yuksek_skor:
                    bulunan_plaka = temiz_plaka
                    en_yuksek_skor = skor

        veriler.append({
            "Dosya Adı": uploaded_file.name,
            "Plaka": bulunan_plaka,
            "Güven Skoru": en_yuksek_skor
        })

        progress_bar.progress((i + 1) / toplam)

    df = pd.DataFrame(veriler)

    st.success("İşlem tamamlandı!")
    st.dataframe(df)

    excel_file = "plaka_sonuclari.xlsx"
    df.to_excel(excel_file, index=False)

    with open(excel_file, "rb") as f:
        st.download_button(
            label="📥 Excel İndir",
            data=f,
            file_name=excel_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
