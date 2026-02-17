import streamlit as st
import re
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Plaka Tanıma Sistemi", layout="wide")

st.title("🚗 Toplu Plaka Tanıma Sistemi")
st.write("Fotoğrafları yükleyin, plakalar otomatik tespit edilsin.")

# ----------------------------
# OCR MODELİ (CACHE'Lİ)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        lang='en',
        show_log=False
    )

ocr = load_ocr()

# ----------------------------
# PLAKA KONTROL FONKSİYONU
# ----------------------------
def plaka_kontrol(metin):
    temiz_metin = re.sub(r'[^A-Z0-9]', '', metin.upper())
    model = r"^\d{2}[A-Z]{1,3}\d{2,5}$"
    if re.match(model, temiz_metin):
        return True, temiz_metin
    return False, None


# ----------------------------
# DOSYA YÜKLEME
# ----------------------------
uploaded_files = st.file_uploader(
    "📂 Fotoğrafları seçin",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:

    veriler = []
    progress_bar = st.progress(0)
    toplam = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        bulunan_plaka = "TESPIT_EDILEMEDI"
        en_yuksek_skor = 0.0

        try:
            result = ocr.ocr(image_np, cls=True)

            if result and result[0]:
                for line in result[0]:
                    metin = line[1][0]
                    skor = line[1][1]

                    yasakli_kelimeler = [
                        "OTOPARK", "TURKIYE", "TR",
                        "ISTANBUL", "BURSA",
                        "CADDE", "SOKAK"
                    ]

                    if any(y in metin.upper() for y in yasakli_kelimeler):
                        continue

                    uygun_mu, temiz_plaka = plaka_kontrol(metin)

                    if uygun_mu and skor > en_yuksek_skor:
                        bulunan_plaka = temiz_plaka
                        en_yuksek_skor = skor

        except Exception as e:
            bulunan_plaka = "HATA"
            en_yuksek_skor = 0.0

        veriler.append({
            "Dosya Adı": uploaded_file.name,
            "Plaka": bulunan_plaka,
            "Güven Skoru": round(en_yuksek_skor, 3)
        })

        progress_bar.progress((i + 1) / toplam)

    df = pd.DataFrame(veriler)

    st.success("✅ İşlem tamamlandı!")
    st.dataframe(df, use_container_width=True)

    # Excel'i RAM üzerinden oluştur
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)

    st.download_button(
        label="📥 Excel İndir",
        data=output,
        file_name="plaka_sonuclari.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
