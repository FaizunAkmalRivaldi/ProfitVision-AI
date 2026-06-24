import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="🚀 ProfitVision AI",
    page_icon="📈",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

h1 {
    text-align: center;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# JUDUL APLIKASI
# =====================================

st.title("🚀 ProfitVision AI")
st.subheader("Smart Policy Simulator & What-If Analysis")

st.write("""
Simulator ini digunakan untuk menganalisis dampak perubahan
anggaran iklan dan diskon terhadap keuntungan toko menggunakan
Machine Learning dan simulasi kebijakan What-If.
""")

# =====================================
# DATA HISTORIS
# =====================================

X_train = np.array([
    [5, 10],
    [10, 20],
    [15, 5],
    [20, 25],
    [25, 15]
])

y_train = np.array([
    50,
    80,
    110,
    90,
    150
])

# =====================================
# TRAINING MODEL
# =====================================

model = LinearRegression()
model.fit(X_train, y_train)

# =====================================
# BASELINE
# =====================================

baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]

# =====================================
# FUNGSI SIMULASI
# =====================================

def run_simulation(new_iklan, new_diskon):

    intervention_input = np.array([
        [new_iklan, new_diskon]
    ])

    prediction = model.predict(intervention_input)[0]

    delta_y = prediction - baseline_pred

    return prediction, delta_y

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("🎛 Variabel Kontrol")

iklan_slider = st.sidebar.slider(
    "Anggaran Iklan (Juta)",
    0,
    50,
    10
)

diskon_slider = st.sidebar.slider(
    "Besaran Diskon (%)",
    0,
    50,
    10
)

# =====================================
# MENJALANKAN SIMULASI
# =====================================

hasil_prediksi, delta = run_simulation(
    iklan_slider,
    diskon_slider
)

# =====================================
# METRIC
# =====================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="💰 Prediksi Keuntungan",
        value=f"Rp {hasil_prediksi:.2f} Juta",
        delta=f"{delta:.2f} Juta"
    )

with col2:
    st.metric(
        label="📊 Keuntungan Baseline",
        value=f"Rp {baseline_pred:.2f} Juta"
    )

# =====================================
# AI POLICY ADVISOR
# =====================================

st.subheader("🤖 AI Policy Advisor")

if delta > 20:
    st.success(
        "🟢 Sangat Direkomendasikan.\n\n"
        "Intervensi ini meningkatkan keuntungan secara signifikan."
    )

elif delta > 0:
    st.info(
        "🟡 Layak Dipertimbangkan.\n\n"
        "Keuntungan meningkat namun belum terlalu besar."
    )

elif delta == 0:
    st.warning(
        "⚪ Tidak Ada Perubahan Signifikan."
    )

else:
    st.error(
        "🔴 Tidak Direkomendasikan.\n\n"
        "Skenario ini menurunkan keuntungan dibanding baseline."
    )

# =====================================
# ANALISIS DELTA
# =====================================

st.subheader("📈 Delta Analysis")

if delta > 0:
    st.success(
        f"Keuntungan meningkat sebesar Rp {delta:.2f} Juta."
    )

elif delta < 0:
    st.error(
        f"Keuntungan menurun sebesar Rp {abs(delta):.2f} Juta."
    )

else:
    st.info(
        "Tidak terjadi perubahan terhadap baseline."
    )

# =====================================
# GAUGE CHART
# =====================================

st.subheader("🎯 Profit Score Meter")

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=hasil_prediksi,
        title={"text": "Skor Keuntungan"},
        gauge={
            "axis": {"range": [0, 200]}
        }
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# BAR CHART
# =====================================

st.subheader("📊 Perbandingan Baseline vs Intervensi")

data_plot = pd.DataFrame({
    "Skenario": [
        "Baseline",
        "Intervensi"
    ],
    "Keuntungan": [
        baseline_pred,
        hasil_prediksi
    ]
})

st.bar_chart(
    data=data_plot,
    x="Skenario",
    y="Keuntungan"
)

# =====================================
# TOP 5 SKENARIO TERBAIK
# =====================================

hasil = []

for iklan in range(0, 51, 5):
    for diskon in range(0, 51, 5):

        prediksi = model.predict(
            [[iklan, diskon]]
        )[0]

        hasil.append([
            iklan,
            diskon,
            prediksi
        ])

df_top = pd.DataFrame(
    hasil,
    columns=[
        "Iklan",
        "Diskon",
        "Keuntungan"
    ]
)

top5 = df_top.sort_values(
    by="Keuntungan",
    ascending=False
).head(5)

st.subheader("🏆 Top 5 Skenario Terbaik")

st.dataframe(
    top5,
    use_container_width=True
)

# =====================================
# TABEL HASIL
# =====================================

st.subheader("📋 Ringkasan Simulasi")

hasil_df = pd.DataFrame({
    "Parameter": [
        "Anggaran Iklan",
        "Diskon",
        "Keuntungan Baseline",
        "Keuntungan Intervensi",
        "Delta"
    ],
    "Nilai": [
        f"{iklan_slider} Juta",
        f"{diskon_slider} %",
        f"{baseline_pred:.2f} Juta",
        f"{hasil_prediksi:.2f} Juta",
        f"{delta:.2f} Juta"
    ]
})

st.dataframe(
    hasil_df,
    use_container_width=True
)

# =====================================
# FOOTER
# =====================================

st.markdown("---")
st.caption(
    "ProfitVision AI © 2026 | Simulator Kebijakan Berbasis Machine Learning & What-If Analysis"
)