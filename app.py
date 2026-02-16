import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

# Sayfa ayarı
st.set_page_config(page_title="Otopark Plaka Tanıma", layout="wide")

st.title("☁️ Bulut Otopark Sistemi")
st.info("Bu sistem 7/24 Aktiftir. Bilgisayar kapalıyken de çalışır.")

# OCR Modelini Yükle (Önbelleğe al ki her seferinde yüklemesin)
@st.cache_resource
def load_model():
    # Lang='en' bazen plakalarda 'tr'den daha iyi sonuç verir, deneyebilirsin.
    return PaddleOCR(use_angle_cls=True, lang='en')

try:
    ocr_model = load_model()
    st.success("✅ OCR Motoru Hazır!")
except Exception as e:
    st.error(f"OCR Modeli yüklenirken hata oluştu: {e}")

# Otopark Seçimi
otoparklar = ["Kadıköy", "Beşiktaş", "Nişantaşı"]
secim = st.selectbox("Lokasyon Seç:", otoparklar)

# Fotoğraf Yükleme Alanı
dosyalar = st.file_uploader("Fotoğrafları Yükle (Çoklu Seçim)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if st.button("Analizi Başlat") and dosyalar:
    st.write(f"🔍 {len(dosyalar)} adet fotoğraf taranıyor...")
    
    # Sonuçları göstermek için bir tablo/liste yapısı
    sonuclar = []

    for dosya in dosyalar:
        # 1. Dosyayı OpenCV formatına çevir
        file_bytes = np.asarray(bytearray(dosya.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # 2. OCR İşlemi
        result = ocr_model.ocr(img, cls=True)

        # 3. Sonucu Yakala
        plaka_metni = "Bulunamadı"
        if result and result[0]:
            # En yüksek güven oranına sahip metni alalım
            txts = [line[1][0] for line in result[0]]
            plaka_metni = ", ".join(txts)
        
        sonuclar.append({"Dosya": dosya.name, "Okunan": plaka_metni})
        
        # Ekrana bas (İstersen kapatabilirsin)
        with st.expander(f"📸 {dosya.name} -> {plaka_metni}"):
            st.image(dosya, width=300)

    st.success("İşlem Tamamlandı!")
    st.table(sonuclar)

elif st.button("Analizi Başlat") and not dosyalar:
    st.warning("Lütfen önce fotoğraf seçin.")