import os
# 1. KRİTİK AYAR: Hızlandırma modunu (MKLDNN) zorla kapatıyoruz.
# Bu satırlar importlardan ÖNCE gelmeli.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0" 

import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

# Sayfa ayarı
st.set_page_config(page_title="Otopark Plaka Tanıma", layout="wide")

st.title("☁️ Bulut Otopark Sistemi")
st.info("Bu sistem 7/24 Aktiftir.")

# OCR Modelini Yükle
@st.cache_resource
def load_model():
    # enable_mkldnn=False diyerek hatayı kökten çözüyoruz.
    # use_angle_cls=False ile gereksiz açı kontrolünü kapatıp hız kazanıyoruz.
    return PaddleOCR(lang='en', enable_mkldnn=False, use_angle_cls=False)

try:
    with st.spinner("Sistem Hazırlanıyor..."):
        ocr_model = load_model()
    st.success("✅ Sistem Hazır!")
except Exception as e:
    st.error(f"Model yüklenirken hata: {e}")

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
                    # cls=True parametresini kaldırdık, düz okuma yapacak.
                    result = ocr_model.ocr(img, cls=False)

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
