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
    # Model yüklenirken hata almamak için temel ayarlar
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

try:
    with st.spinner("OCR Motoru Hazırlanıyor..."):
        ocr_model = load_model()
    st.success("✅ OCR Motoru Hazır!")
except Exception as e:
    st.error(f"OCR Modeli yüklenirken hata oluştu: {e}")

# Otopark Seçimi
otoparklar = ["Kadıköy", "Beşiktaş", "Nişantaşı"]
secim = st.selectbox("Lokasyon Seç:", otoparklar)

# Fotoğraf Yükleme Alanı
dosyalar = st.file_uploader("Fotoğrafları Yükle (Çoklu Seçim)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

# --- DÜZELTİLMİŞ TEK BUTON ALANI ---
if st.button("Analizi Başlat"):
    if dosyalar:
        st.write(f"🔍 {len(dosyalar)} adet fotoğraf taranıyor...")
        
        # Sonuçları tutacak liste
        sonuclar = []
        
        # İlerleme çubuğu
        bar = st.progress(0)
        
        for i, dosya in enumerate(dosyalar):
            try:
                # 1. Dosyayı OpenCV formatına çevir
                file_bytes = np.asarray(bytearray(dosya.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)

                # 2. OCR İşlemi
                if img is not None:
                    result = ocr_model.ocr(img, cls=True)

                    # 3. Sonucu Yakala
                    plaka_metni = "Okunamadı"
                    if result and result[0]:
                        # En güvenilir metinleri al
                        txts = [line[1][0] for line in result[0]]
                        plaka_metni = ", ".join(txts)
                    
                    sonuclar.append({"Dosya": dosya.name, "Okunan": plaka_metni})
                    
                    # Sonucu göster (Açılır kapanır kutu içinde)
                    with st.expander(f"📸 {dosya.name} -> {plaka_metni}"):
                        st.image(dosya, width=300)
                else:
                    st.error(f"{dosya.name} dosyası okunamadı.")
            
            except Exception as e:
                st.error(f"Hata ({dosya.name}): {e}")

            # İlerleme çubuğunu güncelle
            bar.progress((i + 1) / len(dosyalar))

        # Tüm işlem bitince tabloyu göster
        st.success("✅ İşlem Tamamlandı!")
        if sonuclar:
            st.table(sonuclar)

    else:
        # Dosya seçilmediyse uyarı ver
        st.warning("⚠️ Lütfen önce fotoğraf yükleyin.")
