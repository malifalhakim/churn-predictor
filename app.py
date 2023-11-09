import streamlit as st
from prediction import predict_churn,show_shap_value
import numpy as np
st.set_page_config(page_title="Churn Prediction")

# Style External
st.markdown('<style>.stButton{margin-top:30px;display:flex;justify-content:center;margin-bottom:20px;}</style>', unsafe_allow_html=True)

# Judul
st.markdown("<h1 style='text-align: center'>Churn Prediction</h1>", unsafe_allow_html=True)

# Latar Belakang
st.markdown("<div style='margin-top:50px;margin-bottom:60px;border-style:dotted;padding:10px;text-align:justify'><em>Churn</em> merujuk pada fenomena ketika pelanggan atau konsumen berhenti menggunakan layanan atau produk yang ditawarkan oleh suatu perusahaan. Hal ini menjadi perhatian utama bagi banyak bisnis karena dapat berdampak negatif pada pendapatan dan pertumbuhan perusahaan. Mengidentifikasi potensi churn menjadi krusial karena mencegah kehilangan pelanggan lebih murah daripada mencari pelanggan baru. Model ini dapat menganalisis perilaku pelanggan berdasarkan data historis dan mengidentifikasi calon pelanggan yang berpotensi untuk churn. Dengan demikian, perusahaan dapat mengambil langkah-langkah preventif atau strategis untuk mempertahankan pelanggan yang rentan untuk churn melalui tindakan yang dapat mempengaruhi keputusan pelanggan atau meningkatkan layanan produk.</div>",unsafe_allow_html=True)

# Form
# Fungsi untuk membuat selectbox
def create_selectbox(field_name, field_options):
    selected_value = st.selectbox(f"{field_name}", list(field_options.keys()))
    return selected_value

# Dictionary yang berisi field dan nilai-nilai yang sesuai
fields = {
    'Location': {'Jakarta': 0, 'Bandung': 1},
    'Device Class': {'Low End': 0, 'Mid End': 1, 'High End': 2},
    'Games Product': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Music Product': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Education Product': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Call Center': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Video Product': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Use MyApp': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Payment Method': {'Pulsa': 0, 'Digital Wallet': 1, 'Debit': 2, 'Credit': 3}
}

col1, col2 = st.columns(2)

with col1:
    tenure_months = st.number_input(label="Tenure Months", min_value=0, step=1)
    tenure_months = int(tenure_months)

# Membuat selectbox untuk setiap field
selected_values = {}
is_col1 = False
for field, values in fields.items():
    if is_col1:
        with col1:
            selected_values[field] = create_selectbox(field, values)
        is_col1 = False
    else:
        with col2:
            selected_values[field] = create_selectbox(field,values)
        is_col1 = True

with col1:
    monthly_purchase = st.number_input(label="Monthly Purchase", min_value=0.0,step=0.00001,format="%f")
    monthly_purchase = float(monthly_purchase)

# Mendapatkan nilai-nilai dari setiap field secara berurutan
ordered_values = []
for field, value in selected_values.items():
    ordered_values.append(fields[field][value])

ordered_values = [tenure_months] + ordered_values + [monthly_purchase]
data = np.array(ordered_values)

# Button untuk eksekusi
if st.button("Predict"):
    with st.spinner("Predicting...."):
        st.markdown("<h6>Churn Prediction :</h6>", unsafe_allow_html=True)
        # TO DO : UBAH BAGIAN INI
        pred = predict_churn(data)[1]
        if pred > 0.5:
            st.error(f"Didapati probabilitas sebesar {pred*100:.2f}% bahwa pelanggan akan berhenti berlangganan.")
        else:
            st.warning(f"Didapati probabilitas sebesar {pred*100:.2f}% bahwa pelanggan akan berhenti berlangganan.")

        show_shap_value(data)

        with st.expander("Cara membaca grafik"):
            st.markdown("<div style='text-align:justify;padding:20px'> Grafik diatas menunjukkan setiap fitur dan kontribusinya terhadap output dari model. Jika output dari model semakin tinggi, maka semakin tinggi pula probabilitas pelanggan untuk berhenti berlangganan. Fitur dengan bar chart berwarna merah menandakan bahwa fitur tersebut berkontribusi dalam meningkatkan output dari model (probabilitas churn). Sementara, fitur dengan bar chart berwarna biru menandakan bahwa fitur tersebut berkontribusi dalam menurunkan output dari model (probabilitas churn).</div>",unsafe_allow_html=True)