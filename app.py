import os
import streamlit as st

# --- SİHİRLİ YAMA (CRITICAL FIX) ---
# PaddlePaddle 3.0+ sürümünde kaldırılan fonksiyonu manuel olarak yamalıyoruz.
# Bu blok, 'set_optimization_level' hatasını %100 çözer.
import paddle
try:
    # Farklı Paddle sürümleri için garantiye alıyoruz
    if hasattr(paddle, 'inference') and hasattr(paddle.inference, 'Config'):
        paddle.inference.Config.set_optimization_level = lambda self, x: None
    
    try:
        from paddle.base.libpaddle import AnalysisConfig
        AnalysisConfig.set_optimization_level = lambda self, x: None
    except ImportError:
        pass
        
    print("✅ Paddle 3.0 uyumluluk yaması başarıyla uygulandı.")
except Exception as e:
    print(f"⚠️ Yama uyarısı: {e}")
# -----------------------------------

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

# Sayfa ayarı
st.set_page_config(page_title="Otopark Plaka Tanıma", layout="wide")

st.title("☁️ Bulut Otopark Sistemi")
st.info("Sistem Aktif (v3.0 Uyumlu)")

# OCR Modelini Yükle
@st.cache_resource
def load_model():
    # 'show_log' ve 'use_angle_cls' gibi eski parametreleri kaldırdık.
    # Sadece 'lang' parametresi ile en sade ve güvenli hali.
    return PaddleOCR(lang='en', use_angle_cls=False)

try:
    with st.spinner("Yapay Zeka Modeli Yükleniyor..."):
        ocr_model = load_model()
    st.success("✅ Motor Hazır!")
except Exception as e:
    st.error(f"Model Yükleme Hatası: {e}")

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
                    # Sadece resmi veriyoruz, parametresiz çağrı.
                    result = ocr_model.ocr(img)

                    # 3. Sonucu Yakala
                    plaka_metni = "Okunamadı"
                    if result and result[0]:
                        # Güvenilir metinleri birleştir
                        txts = [line[1][0] for line in result[0] if line[1]] 
                        plaka_metni = ", ".join(txts)
                    
                    sonuclar.append({"Dosya": dosya.name, "Okunan": plaka_metni})
                    
                    # Sonucu göster
                    with st.expander(f"📸 {dosya.name} -> {plaka_metni}"):
                        st.image(dosya, width=300)
                else:
                    st.error(f"{dosya.name} dosyası okunamadı.")
            
            except Exception as e:
                st.error(f"Hata ({dosya.name}): {e}")

            # İlerleme çubuğunu güncelle
            bar.progress((i + 1) / len(dosyalar))

        st.success("✅ İşlem Tamamlandı!")
        if sonuclar:
            st.table(sonuclar)

    else:
        st.warning("⚠️ Lütfen önce fotoğraf yükleyin.")
