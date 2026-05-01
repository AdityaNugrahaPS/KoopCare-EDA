import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Analisis Pembiayaan",
    page_icon="🕌",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f9f9f6; }
    .stApp { font-family: 'Segoe UI', sans-serif; }

    .header-box {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .header-box h1 { font-size: 1.8rem; margin: 0; }
    .header-box p  { font-size: 0.95rem; opacity: 0.85; margin: 0.4rem 0 0; }

    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1a5276;
        border-left: 4px solid #2e86c1;
        padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem;
    }

    .result-layak {
        background: linear-gradient(135deg, #1e8449, #27ae60);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-review {
        background: linear-gradient(135deg, #b7770d, #f39c12);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-tolak {
        background: linear-gradient(135deg, #922b21, #e74c3c);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-layak h2, .result-review h2, .result-tolak h2 {
        font-size: 1.8rem; margin: 0;
    }
    .result-layak p, .result-review p, .result-tolak p {
        font-size: 0.95rem; opacity: 0.9; margin: 0.3rem 0 0;
    }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #1a5276; }
    .metric-card .lbl { font-size: 0.8rem; color: #777; margin-top: 0.2rem; }

    .info-box {
        background: #eaf4fb;
        border-left: 4px solid #2e86c1;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #1a5276;
        margin-top: 1rem;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("best_model_abel.pkl")

try:
    obj       = load_model()
    model     = obj["model"]
    threshold = obj["threshold"]
    feat_cols = obj["feature_cols"]
except:
    st.error("Model tidak ditemukan. Pastikan best_model_abel.pkl ada di folder yang sama.")
    st.stop()

# ── Feature engineering ───────────────────────────────────────────────────────
def feature_engineering(df):
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"]      = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["AGE_YEARS"]          = abs(df["DAYS_BIRTH"]) / 365
    df["DEBT_TO_INCOME"]     = df["AMT_CREDIT"]  / (df["AMT_INCOME_TOTAL"] + 1)
    df["PAYMENT_RATE"]       = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
    df["EXT_SOURCE_MIN"]  = df[ext].min(axis=1)
    df["EXT_SOURCE_PROD"] = df[ext].prod(axis=1)
    return df

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🕌 Sistem Analisis Pembiayaan Syariah</h1>
    <p>Analisis kelayakan pembiayaan berbasis Machine Learning · BMT / Koperasi Syariah</p>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("form_pembiayaan"):

    # ── Data Pribadi ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">👤 Data Pribadi</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        nama         = st.text_input("Nama Lengkap")
        gender       = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)
        tgl_lahir    = st.date_input("Tanggal Lahir", value=date(1990, 1, 1),
                                     min_value=date(1940, 1, 1), max_value=date(2005, 12, 31))
    with col2:
        status_nikah = st.selectbox("Status Pernikahan", [
            "Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
        jml_anak     = st.number_input("Jumlah Anak", min_value=0, max_value=20, value=0)
        jml_kk       = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=20, value=2)

    # ── Pekerjaan & Penghasilan ───────────────────────────────────────────────
    st.markdown('<div class="section-title">💼 Pekerjaan & Penghasilan</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        jenis_kerja  = st.selectbox("Jenis Pekerjaan", [
            "Working", "Commercial associate", "Pensioner",
            "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"])
        pendidikan   = st.selectbox("Pendidikan Terakhir", [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"])
    with col4:
        profesi      = st.selectbox("Profesi / Jabatan", [
            "Laborers", "Core staff", "Managers", "Drivers", "Sales staff",
            "Accountants", "Medicine staff", "Cooking staff", "Security staff",
            "Cleaning staff", "Private service staff", "Low-skill Laborers",
            "Waiters/barmen staff", "Secretaries", "HR staff", "Realty agents",
            "IT staff"])
        lama_kerja   = st.number_input("Lama Bekerja (tahun)", min_value=0.0, max_value=50.0,
                                       value=3.0, step=0.5)
        penghasilan  = st.number_input("Penghasilan per Bulan (Rp)", min_value=0,
                                       value=5000000, step=500000)

    # ── Kepemilikan Aset ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏠 Kepemilikan Aset</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        punya_motor = st.radio("Punya Kendaraan?", ["Ya", "Tidak"], horizontal=True)
    with col6:
        punya_rumah = st.radio("Punya Properti?", ["Ya", "Tidak"], horizontal=True)

    # ── Detail Pembiayaan ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💰 Detail Pembiayaan</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        jml_pinjaman = st.number_input("Jumlah Pembiayaan (Rp)", min_value=0,
                                       value=10000000, step=1000000)
        harga_barang = st.number_input("Harga Barang / Tujuan (Rp)", min_value=0,
                                       value=9000000, step=500000)
    with col8:
        cicilan      = st.number_input("Cicilan per Bulan (Rp)", min_value=0,
                                       value=500000, step=100000)
        ganti_hp     = st.number_input("Terakhir Ganti Nomor HP (bulan lalu)",
                                       min_value=0, max_value=120, value=6)

    # ── Skor Eksternal ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Skor Biro Kredit</div>', unsafe_allow_html=True)
    st.caption("Diisi oleh sistem / petugas dari data biro kredit (0.0 – 1.0)")
    col9, col10, col11 = st.columns(3)
    with col9:
        ext1 = st.number_input("Skor Biro 1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    with col10:
        ext2 = st.number_input("Skor Biro 2", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    with col11:
        ext3 = st.number_input("Skor Biro 3", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    submitted = st.form_submit_button("🔍 Analisis Kelayakan")

# ── Prediksi ──────────────────────────────────────────────────────────────────
if submitted:
    today     = date.today()
    days_birth     = -abs((today - tgl_lahir).days)
    days_employed  = -abs(int(lama_kerja * 365)) if lama_kerja > 0 else 365243
    days_phone     = -abs(ganti_hp * 30)

    input_data = {
        "CODE_GENDER"          : "M" if gender == "Laki-laki" else "F",
        "NAME_INCOME_TYPE"     : jenis_kerja,
        "NAME_EDUCATION_TYPE"  : pendidikan,
        "NAME_FAMILY_STATUS"   : status_nikah,
        "OCCUPATION_TYPE"      : profesi,
        "FLAG_OWN_CAR"         : "Y" if punya_motor == "Ya" else "N",
        "FLAG_OWN_REALTY"      : "Y" if punya_rumah == "Ya" else "N",
        "CNT_CHILDREN"         : jml_anak,
        "CNT_FAM_MEMBERS"      : float(jml_kk),
        "AMT_INCOME_TOTAL"     : float(penghasilan),
        "AMT_CREDIT"           : float(jml_pinjaman),
        "AMT_ANNUITY"          : float(cicilan),
        "AMT_GOODS_PRICE"      : float(harga_barang),
        "DAYS_EMPLOYED"        : float(days_employed),
        "DAYS_LAST_PHONE_CHANGE": float(days_phone),
        "DAYS_BIRTH"           : float(days_birth),
        "EXT_SOURCE_1"         : ext1,
        "EXT_SOURCE_2"         : ext2,
        "EXT_SOURCE_3"         : ext3,
    }

    df_input = pd.DataFrame([input_data])
    df_fe    = feature_engineering(df_input)
    X        = df_fe[feat_cols]

    prob  = model.predict_proba(X)[:, 1][0]
    skor  = round((1 - prob) * 100, 1)
    persen = round(prob * 100, 1)

    st.markdown("---")
    st.markdown("### Hasil Analisis")

    # Zona keputusan
    if prob < 0.40:
        st.markdown(f"""
        <div class="result-layak">
            <h2>✅ LAYAK DIBIAYAI</h2>
            <p>Nasabah memenuhi kriteria kelayakan pembiayaan</p>
        </div>""", unsafe_allow_html=True)
    elif prob < threshold:
        st.markdown(f"""
        <div class="result-review">
            <h2>🟡 PERLU REVIEW</h2>
            <p>Diperlukan verifikasi lebih lanjut oleh petugas</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-tolak">
            <h2>❌ TIDAK LAYAK</h2>
            <p>Risiko gagal bayar terlalu tinggi</p>
        </div>""", unsafe_allow_html=True)

    # Metrik
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="val">{skor}</div>
            <div class="lbl">Skor Kredit (0–100)</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="val">{persen}%</div>
            <div class="lbl">Risiko Gagal Bayar</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        zona = "Rendah 🟢" if prob < 0.40 else ("Sedang 🟡" if prob < threshold else "Tinggi 🔴")
        st.markdown(f"""
        <div class="metric-card">
            <div class="val" style="font-size:1.2rem">{zona}</div>
            <div class="lbl">Kategori Risiko</div>
        </div>""", unsafe_allow_html=True)

    # Info zona
    st.markdown("""
    <div class="info-box">
        <b>Keterangan Zona:</b><br>
        🟢 <b>Layak</b> — Risiko &lt; 40% · Otomatis disetujui<br>
        🟡 <b>Perlu Review</b> — Risiko 40%–68% · Verifikasi petugas<br>
        🔴 <b>Tidak Layak</b> — Risiko ≥ 68% · Ditolak
    </div>
    """, unsafe_allow_html=True)

    # Detail input
    with st.expander("📋 Detail Input"):
        st.write(f"**Nama:** {nama if nama else '-'}")
        st.write(f"**Usia:** {abs(days_birth) // 365} tahun")
        st.write(f"**Rasio Cicilan/Penghasilan:** {round(cicilan/penghasilan*100, 1) if penghasilan > 0 else '-'}%")
        st.write(f"**Rasio Kredit/Penghasilan:** {round(jml_pinjaman/penghasilan, 1) if penghasilan > 0 else '-'}x")
        st.dataframe(df_input.T.rename(columns={0: "Nilai"}))
