import streamlit as st
from prediction import predict_churn,predict_cltv,show_shap_value,show_shap_value_reg
import numpy as np
import shap
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Churn and CLTV Prediction')

tenure_months = st.number_input(label="Tenure Months", min_value=0, step=1)
tenure_months = int(tenure_months)

# Fungsi untuk membuat selectbox
def create_selectbox(field_name, field_options):
    selected_value = st.selectbox(f"{field_name}", list(field_options.keys()))
    return selected_value

# Dictionary yang berisi field dan nilai-nilai yang sesuai
fields = {
    'Location': {'Jakarta-0': 0, 'Bandung-1': 1},
    'Device Class': {'Low End-0': 0, 'Mid End-1': 1, 'High End-2': 2},
    'Games Product': {'No-0': 0, 'Yes-1': 1, 'No internet service-2': 2},
    'Music Product': {'No-0': 0, 'Yes-1': 1, 'No internet service-2': 2},
    'Education Product': {'No-0': 0, 'Yes-1': 1, 'No internet service-2': 2},
    'Call Center': {'No-0': 0, 'Yes-1': 1, 'No internet service-2': 2},
    'Video Product': {'No-0': 0, 'Yes-1': 1, 'No internet service-2': 2},
    'Use MyApp': {'No-0': 0, 'Yes-1': 1, 'No internet service-2': 2},
    'Payment Method': {'Pulsa-0': 0, 'Digital Wallet-1': 1, 'Debit-2': 2, 'Credit-3': 3}
}

# Membuat selectbox untuk setiap field
selected_values = {}
for field, values in fields.items():
    selected_values[field] = create_selectbox(field, values)

monthly_purchase = st.number_input(label="Monthly Purchase", min_value=0.0,step=0.0000001)
monthly_purchase = float(monthly_purchase)

# Mendapatkan nilai-nilai dari setiap field secara berurutan
ordered_values = []
for field, value in selected_values.items():
    ordered_values.append(fields[field][value])

ordered_values = [tenure_months] + ordered_values + [monthly_purchase]
data = np.array(ordered_values)

# Button untuk eksekusi
if st.button("Proses"):
    st.write("Churn-Prediction")
    st.text(predict_churn(np.append(data,tenure_months*monthly_purchase)))
    show_shap_value(np.append(data,tenure_months*monthly_purchase))
    st.write("CLTV-Prediction")
    st.text(predict_cltv(data)[0])
    show_shap_value_reg(data)