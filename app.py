import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

# Sayfa ayarı
st.set_page_config(page_title="Otopark Plaka Tanıma", layout="wide")

st.title("☁️ Bulut Otopark Sistemi")
st.info("Sistem Hazır! (EasyOCR Modu)")

# OCR Modelini Yükle (Önbelleğe alıyoruz)
@st.cache_resource
def load_model():
    # 'en' parametresi İngilizce karakterler (plakalar) için yeterlidir.
    # gpu=False diyerek sunucuda sadece işlemci kullanmasını sağlıyoruz.
    return easyocr.Reader(['en'], gpu=False)

try:
    with st.spinner("OCR Modeli Yükleniyor..."):
        reader = load_model()
    st.success("✅ Sistem Çalışıyor!")
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
                    # EasyOCR okuması
                    results = reader.readtext(img)

                    # 3. Sonucu Yakala
                    plaka_metni = "Okunamadı"
                    if results:
                        # EasyOCR çıktısı: (bbox, text, prob)
                        # Biz sadece text kısmını alıp birleştiriyoruz.
                        bulunanlar = [res[1] for res in results if res[2] > 0.2] # %20 üzeri güvenilirlik
                        plaka_metni = ", ".join(bulunanlar)
                    
                    sonuclar.append({"Dosya": dosya.name, "Okunan": plaka_metni})
                    
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
