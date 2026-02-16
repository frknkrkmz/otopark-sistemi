import os
import streamlit as st

# --- SİHİRLİ YAMA (MONKEY PATCH) ---
# PaddlePaddle 3.0+ sürümünde kaldırılan 'set_optimization_level' fonksiyonunu
# manuel olarak ekliyoruz ki PaddleOCR hata vermesin.
import paddle
try:
    from paddle.base.libpaddle import AnalysisConfig
    if not hasattr(AnalysisConfig, 'set_optimization_level'):
        AnalysisConfig.set_optimization_level = lambda self, x: None
        print("✅ Paddle 3.0 uyumluluk yaması uygulandı.")
except Exception as e:
    print(f"⚠️ Yama uygulanamadı: {e}")
# -----------------------------------

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

# Sayfa ayarı
st.set_page_config(page_title="Otopark Plaka Tanıma", layout="wide")

st.title("☁️ Bulut Otopark Sistemi")
st.info("Sistem Hazır! (Paddle 3.0 Uyumlu)")

# OCR Modelini Yükle
@st.cache_resource
def load_model():
    # mkldnn kapatıyoruz (Hızlandırma hatasını önlemek için)
    return PaddleOCR(lang='en', use_angle_cls=False, enable_mkldnn=False)

try:
    with st.spinner("Sistem Hazırlanıyor..."):
        ocr_model = load_model()
    st.success("✅ Motor Çalışıyor!")
except Exception as e:
    st.error(f"Kritik Hata: {e}")

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
                    # Sadece resmi veriyoruz
                    result = ocr_model.ocr(img)

                    # 3. Sonucu Yakala
                    plaka_metni = "Okunamadı"
                    if result and result[0]:
                        # En güvenilir metinleri al
                        txts = [line[1][0] for line in result[0] if line[1]] 
                        plaka_metni = ", ".join(txts)
                    
                    sonuclar.append({"Dosya": dosya.name, "Okunan": plaka_metni})
                    
                    # Sonucu göster
                    with st.expander(f"📸 {dosya.name} -> {plaka_metni}"):
                        st.image(dosya, width=300)
                else:
                    st.error(f"{dosya.name} okunamadı.")
            
            except Exception as e:
                st.error(f"Hata ({dosya.name}): {e}")

            # İlerleme çubuğunu güncelle
            bar.progress((i + 1) / len(dosyalar))

        st.success("✅ İşlem Tamamlandı!")
        if sonuclar:
            st.table(sonuclar)

    else:
        st.warning("⚠️ Lütfen önce fotoğraf yükleyin.")
