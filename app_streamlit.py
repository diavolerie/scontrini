import streamlit as st

st.set_page_config(
    page_title="Scontrini PDF",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Scontrini PDF")
st.caption("Versione web — Fase 1: caricamento scansioni, PDF e impostazioni")

with st.sidebar:
    st.header("Scansioni")
    scans = st.file_uploader(
        "Aggiungi scansioni",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
    )

    st.header("PDF di riferimento")
    pdf = st.file_uploader("Carica PDF", type=["pdf"])

    st.header("Impostazioni")
    soglia = st.slider("Soglia riconoscimento crocini", 0.10, 0.95, 0.50, 0.01)

    distanza = st.number_input(
        "Distanza crocini D (px, 0 = automatica)",
        min_value=0.0,
        value=0.0,
        step=10.0,
    )

    rotazione = st.checkbox("Ruota scansioni di 180°")

    formato = st.selectbox("Formato di uscita", ["png", "jpg"])
    padding = st.number_input("Padding nome file", 0, 6, 2)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Scansioni caricate")
    if scans:
        st.success(f"{len(scans)} scansioni caricate")
        for f in scans:
            st.write(f"- `{f.name}` — {f.size / 1024:.0f} KB")
    else:
        st.info("Nessuna scansione caricata.")

with col2:
    st.subheader("PDF")
    if pdf:
        st.success(f"PDF caricato: `{pdf.name}`")
    else:
        st.info("Nessun PDF caricato.")

st.divider()

if st.button("Avvia analisi", type="primary", use_container_width=True):
    if not scans:
        st.error("Carica almeno una scansione.")
    elif not pdf:
        st.warning("Puoi analizzare le scansioni, ma per il riconoscimento automatico serve anche il PDF.")
    else:
        st.info("Analisi non ancora collegata al motore originale: questa è la Fase 1.")

st.subheader("Impostazioni correnti")
st.write(
    f"Soglia: **{soglia:.2f}** · D: **{int(distanza) if distanza else 'automatica'} px** · "
    f"Rotazione: **{'sì' if rotazione else 'no'}** · Formato: **{formato}** · "
    f"Padding: **{padding}**"
)