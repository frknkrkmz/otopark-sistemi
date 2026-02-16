import os

# --- KRİTİK AYARLAR (EN BAŞTA OLmalı) ---
# Paddle 3.0'ın yeni motorunu (PIR) ve MKLDNN hızlandırmayı zorla kapatıyoruz.
# Aldığın "ConvertPirAttribute" hatasının kesin çözümü budur.
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_dn_enable_mkldnn"] = "0"

import streamlit as st
import paddle
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

# --- SİHİRLİ YAMA (MONKEY PATCH) ---
# 'set_optimization_level' hatasını önlemek için koruma kalkanı.
try:
    if hasattr(paddle, 'inference') and hasattr(paddle.inference, 'Config'):
        paddle.inference.Config.set_optimization_level = lambda self, x: None
    
    try:
        from paddle.base.libpaddle import AnalysisConfig
        AnalysisConfig.set_optimization_level = lambda self, x: None
    except ImportError:
        pass
except Exception:
    pass
# -----------------------------------

# Sayfa ayarı
st.set_page_config(page_title="Otopark Plaka Tanıma", layout="wide")

st.title("☁️ Bulut Otopark Sistemi")
st.info("Sistem Hazır! (Güvenli Mod v3.0)")

# OCR Modelini Yükle
@st.cache_resource
def load_model():
    # Tüm hızlandırmaları kapattık, en güvenli modda çalışacak.
    return PaddleOCR(lang='en', use_angle_cls=False, enable_mkldnn=False)

try:
    with st.spinner("Sistem Başlatılıyor..."):
        ocr_model = load_model()
    st.success("✅ Motor Hazır!")
except Exception as e:
    st.error(f"Başlatma Hatası: {e}")

# Otopark Seçimi
otoparklar = ["Kadıköy", "Beşiktaş", "Nişantaşı"]
secim = st.selectbox("Lokasyon Seç:", otoparklar)

# Fotoğraf Yükleme Alanı
dosyalar = st.file_uploader("Fotoğrafları Yükle", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

# --- ANALİZ BUTONU ---
if st.button("Analizi Başlat"):
    if dosyalar:
        st.write(f"🔍 {len(dosyalar)} fotoğraf taranıyor...")
        
        sonuclar = []
        bar = st.progress(0)
        
        for i, dosya in enumerate(dosyalar):
            try:
                # 1. Dosyayı OpenCV formatına çevir
                file_bytes = np.asarray(bytearray(dosya.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)

                # 2. OCR İşlemi
                if img is not None:
                    # Parametresiz sade çağrı
                    result = ocr_model.ocr(img)

                    # 3. Sonucu Yakala
                    plaka_metni = "Okunamadı"
                    if result and result[0]:
                        txts = [line[1][0] for line in result[0] if line[1]] 
                        plaka_metni = ", ".join(txts)
                    
                    sonuclar.append({"Dosya": dosya.name, "Okunan": plaka_metni})
                    
                    with st.expander(f"📸 {dosya.name} -> {plaka_metni}"):
                        st.image(dosya, width=300)
                else:
                    st.error(f"{dosya.name} okunamadı.")
            
            except Exception as e:
                st.error(f"Hata ({dosya.name}): {e}")

            bar.progress((i + 1) / len(dosyalar))

        st.success("✅ İşlem Tamamlandı!")
        if sonuclar:
            st.table(sonuclar)

    else:
        st.warning("⚠️ Lütfen önce fotoğraf yükleyin.")
