import io
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF

st.set_page_config(
    page_title="Scontrini PDF",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Stile simile all'app originale: crema, pannelli chiari, accenti arancio/verde
st.markdown(
    """
    <style>
      .stApp {
          background-color: #E6DFCB;
          color: #1F2A24;
      }
      section[data-testid="stSidebar"] {
          background-color: #EEE8D6;
          border-right: 1px solid #B9B19A;
      }
      h1, h2, h3 {
          color: #1F2A24;
      }
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
          border-color: #2A2A27;
      }
      .stSuccess {
          background-color: #0B5B43;
          color: #F3EFE1;
      }
      .stInfo {
          background-color: #F7D9A0;
          color: #1F2A24;
      }
      .stError {
          background-color: #E8405A;
          color: #FFFFFF;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧾 Scontrini PDF")
st.caption("Versione web — Fase 2: caricamento, anteprime e preparazione analisi")

with st.sidebar:
    st.header("📁 Scansioni")
    scans = st.file_uploader(
        "Aggiungi scansioni",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
        help="Puoi selezionare più immagini contemporaneamente.",
    )

    st.header("📄 PDF di riferimento")
    pdf = st.file_uploader("Carica PDF", type=["pdf"])

    st.header("⚙️ Impostazioni")
    soglia = st.slider(
        "Soglia riconoscimento crocini",
        0.10, 0.95, 0.50, 0.01,
        help="Valori più bassi trovano più crocini, ma aumentano i falsi positivi.",
    )

    distanza = st.number_input(
        "Distanza crocini D (px)",
        min_value=0.0,
        value=0.0,
        step=10.0,
        help="Lascia 0 per farla stimare automaticamente.",
    )

    rotazione = st.checkbox("Ruota scansioni di 180°")
    formato = st.selectbox("Formato di uscita", ["png", "jpg"])
    padding = st.number_input("Padding nome file", 0, 6, 2)

    st.divider()
    st.subheader("Stato")
    if scans and pdf:
        st.success("Pronto per l’analisi")
    elif scans:
        st.info("Scansioni caricate, PDF mancante")
    elif pdf:
        st.info("PDF caricato, scansioni mancanti")
    else:
        st.warning("Carica scansioni e PDF")

st.subheader("Scansioni caricate")

if not scans:
    st.info("Nessuna scansione caricata. Usa il pannello a sinistra.")
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
st.subheader("PDF di riferimento")

if not pdf:
    st.info("Nessun PDF caricato.")
else:
    try:
        doc = fitz.open(stream=pdf.getvalue(), filetype="pdf")
        st.success(f"PDF valido: **{doc.page_count} pagine**")
        st.write(f"File: `{pdf.name}` — {len(pdf.getvalue()) / 1024:.0f} KB")

        if doc.page_count > 0:
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(0.35, 0.35), alpha=False)
            first_page = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            st.image(first_page, caption="Prima pagina del PDF", use_container_width=True)

        doc.close()
    except Exception as ex:
        st.error(f"Impossibile leggere il PDF: {ex}")

st.divider()

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    avvia = st.button(
        "▶️ AVVIA ANALISI",
        type="primary",
        use_container_width=True,
        disabled=not scans,
    )

with col2:
    st.metric("Scansioni", len(scans) if scans else 0)

with col3:
    st.metric("Pagine PDF", doc.page_count if pdf else 0)

if avvia:
    if not scans:
        st.error("Carica almeno una scansione.")
    else:
        with st.status("Analisi in preparazione…", expanded=True) as status:
            st.write("Scansioni caricate correttamente.")
            if pdf:
                st.write("PDF caricato correttamente.")
            else:
                st.write("PDF non caricato: riconoscimento automatico non disponibile.")

            st.write(
                f"Impostazioni: soglia {soglia:.2f}, D "
                f"{'automatica' if distanza == 0 else f'{int(distanza)} px'}, "
                f"rotazione {'attiva' if rotazione else 'disattivata'}."
            )

            st.write("Il motore di riconoscimento crocini sarà collegato nella Fase 3.")
            status.update(label="Analisi pronta per la Fase 3", state="complete")

st.divider()
st.caption("Scontrini PDF — interfaccia web in costruzione")
