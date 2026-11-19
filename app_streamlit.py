import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Scontrini PDF",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background-color: #E6DFCB; color: #1F2A24; }
      section[data-testid="stSidebar"] {
          background-color: #EEE8D6;
          border-right: 1px solid #B9B19A;
      }
      h1, h2, h3 { color: #1F2A24; }
      .stButton > button {
          background-color: #F3EFE1;
          color: #1F2A24;
          border: 1px solid #2A2A27;
          border-radius: 8px;
          font-weight: 600;
      }
      .stButton > button[kind="primary"] {
          background-color: #FF4F12;
          color: #FFFFFF;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧾 Scontrini PDF")
st.caption("Versione web — Fase 3: riconoscimento crocini e ritaglio pagine")

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def read_image(uploaded, rotate=False):
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def find_dark_marks(gray, min_area=40, max_area=2500):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if not (min_area <= area <= max_area):
            continue

        aspect = w / float(h)
        if not (0.65 <= aspect <= 1.45):
            continue

        fill = area / float(w * h)
        if fill < 0.12:
            continue

        marks.append(
            {
                "x": x + w / 2.0,
                "y": y + h / 2.0,
                "w": w,
                "h": h,
                "area": area,
            }
        )

    return marks


def pair_marks(marks, image_height, max_vertical_gap=0.45):
    pairs = []

    for i, top in enumerate(marks):
        for bottom in marks[i + 1:]:
            dx = abs(bottom["x"] - top["x"])
            dy = bottom["y"] - top["y"]

            if dy <= 10:
                continue

            if dx > max(top["w"], bottom["w"]) * 1.5:
                continue

            if dy > image_height * max_vertical_gap:
                continue

            pairs.append(
                {
                    "top": (top["x"], top["y"]),
                    "bottom": (bottom["x"], bottom["y"]),
                    "score": min(top["area"], bottom["area"]),
                }
            )

    pairs.sort(key=lambda p: (p["top"][0], p["top"][1]))
    return pairs


def crop_between_marks(image, pair, margin=18, min_width=180, min_height=180):
    height, width = image.shape[:2]
    x1, y1 = pair["top"]
    x2, y2 = pair["bottom"]

    left = int(max(0, min(x1, x2) - margin))
    right = int(min(width, max(x1, x2) + margin))
    top = int(max(0, min(y1, y2) - margin))
    bottom = int(min(height, max(y1, y2) + margin))

    if right - left < min_width or bottom - top < min_height:
        return None

    return image[top:bottom, left:right].copy()


def encode_image(image, extension):
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError("Impossibile codificare l'immagine")
    return buffer.tobytes()


with st.sidebar:
    st.header("📁 Scansioni")
    scans = st.file_uploader(
        "Aggiungi scansioni",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
    )

    st.header("⚙️ Analisi")
    sensibilita = st.slider(
        "Sensibilità riconoscimento",
        1, 10, 5,
        help="Aumentala se non trova crocini; diminuiscila se trova troppi falsi positivi.",
    )

    rotazione = st.checkbox("Ruota scansioni di 180°")
    formato = st.selectbox("Formato di uscita", ["png", "jpg"])

    st.divider()
    st.subheader("Stato")
    st.success("Motore crocini attivo") if scans else st.warning("Carica le scansioni")

st.subheader("Scansioni caricate")

if not scans:
    st.info("Carica una o più scansioni dal pannello a sinistra.")
else:
    st.success(f"{len(scans)} scansioni caricate")

    cols = st.columns(4)
    for i, uploaded in enumerate(scans):
        with cols[i % 4]:
            try:
                image = Image.open(uploaded)
                st.image(
                    image,
                    caption=f"{uploaded.name}\n{image.width} × {image.height} px",
                    use_container_width=True,
                )
            except Exception as ex:
                st.error(f"Impossibile leggere `{uploaded.name}`: {ex}")

st.divider()

if st.button("▶️ AVVIA ANALISI", type="primary", use_container_width=True):
    if not scans:
        st.error("Carica almeno una scansione.")
    else:
        results = []
        errors = []

        progress = st.progress(0.0, text="Analisi delle scansioni…")

        for index, uploaded in enumerate(scans):
            try:
                image = read_image(uploaded, rotazione)

                if image is None:
                    errors.append(f"{uploaded.name}: immagine non leggibile")
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                marks = find_dark_marks(gray)

                if sensibilita >= 8:
                    marks += find_dark_marks(gray, min_area=25, max_area=4000)

                pairs = pair_marks(marks, image.shape[0])

                if not pairs:
                    errors.append(f"{uploaded.name}: nessun crocino accoppiato")
                    continue

                for number, pair in enumerate(pairs, start=1):
                    crop = crop_between_marks(image, pair)

                    if crop is None:
                        continue

                    results.append(
                        {
                            "scan": uploaded.name,
                            "number": number,
                            "image": crop,
                        }
                    )

            except Exception as ex:
                errors.append(f"{uploaded.name}: {ex}")

            progress.progress((index + 1) / len(scans), text=f"Analizzata {uploaded.name}")

        progress.empty()

        st.subheader("Risultato analisi")

        if not results:
            st.error("Nessun ritaglio trovato. Prova ad aumentare la sensibilità o a ruotare le scansioni.")
        else:
            st.success(f"Trovati {len(results)} ritagli.")

            cols = st.columns(3)

            for i, result in enumerate(results):
                with cols[i % 3]:
                    st.image(
                        cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB),
                        caption=f"{result['scan']} — ritaglio {result['number']}",
                        use_container_width=True,
                    )

            zip_buffer = io.BytesIO()
            extension = ".png" if formato == "png" else ".jpg"

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for i, result in enumerate(results, start=1):
                    base = Path(result["scan"]).stem
                    filename = f"{base}_{i:03d}{extension}"
                    archive.writestr(filename, encode_image(result["image"], extension))

            st.download_button(
                "⬇️ Scarica tutti i ritagli (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="ritagli_scontrini.zip",
                mime="application/zip",
                use_container_width=True,
            )

        if errors:
            with st.expander(f"⚠️ {len(errors)} avvisi o errori"):
                for error in errors:
                    st.write(f"- {error}")

st.divider()
st.caption("Fase 3: riconoscimento crocini e ritaglio. PDF, OCR e editor arriveranno nelle fasi successive.")
