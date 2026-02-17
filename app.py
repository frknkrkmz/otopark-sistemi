import streamlit as st
import easyocr
import cv2
import numpy as np
import re  # Regex kütüphanesi eklendi

# Sayfa ayarı
st.set_page_config(page_title="Otopark Plaka Tanıma", layout="wide")

st.title("☁️ Bulut Otopark Sistemi")
st.info("Sistem Hazır! (Akıllı Plaka Filtresi Aktif)")

# OCR Modelini Yükle
@st.cache_resource
def load_model():
    return easyocr.Reader(['en'], gpu=False)

try:
    with st.spinner("OCR Modeli Yükleniyor..."):
        reader = load_model()
    st.success("✅ Sistem Çalışıyor!")
except Exception as e:
    st.error(f"Model Yükleme Hatası: {e}")

# Türkiye Plaka Regex Kuralı
# 01-81 ile başlar + Harfler + Rakamlar
def plaka_bul(metin_listesi):
    # OCR'dan gelen parça parça metinleri birleştiriyoruz
    birlesik_metin = " ".join(metin_listesi).upper()
    
    # Gereksiz karakterleri temizle (TR yazısı, noktalar vs.)
    temiz_metin = birlesik_metin.replace("TR", "").replace(".", "").replace("-", " ")
    
    # Regex: (İl Kodu) (Harfler) (Rakamlar)
    # Örnek: 16 AEJ 51, 34 AB 1234
    kural = r'\b(0[1-8]|[1-7][0-9]|8[0-1])\s*[A-Z]{1,3}\s*\d{2,4}\b'
    
    match = re.search(kural, temiz_metin)
    if match:
        return match.group(0) # Bulunan plakayı döndür
    else:
        return None # Plaka formatı bulunamadı

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
                    results = reader.readtext(img)

                    plaka_sonuc = "Plaka Bulunamadı"
                    if results:
                        # Tüm okunan metinleri bir listeye al
                        okunanlar = [res[1] for res in results if res[2] > 0.2]
                        
                        # Fonksiyona gönder, sadece plakayı ayıklasın
                        bulunan_plaka = plaka_bul(okunanlar)
                        
                        if bulunan_plaka:
                            plaka_sonuc = bulunan_plaka
                        else:
                            # Eğer formatı yakalayamazsa yine de ham metni gösterelim (Debug için)
                            plaka_sonuc = f"Format Yakalanamadı: {', '.join(okunanlar)}"
                    
                    sonuclar.append({"Dosya": dosya.name, "Plaka": plaka_sonuc})
                    
                    with st.expander(f"📸 {dosya.name} -> {plaka_sonuc}"):
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
