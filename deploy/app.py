import streamlit as st
import pandas as pd
from joblib import load

# Judul Aplikasi
st.title("Prediksi Spesies Penguin")

# Input Data Baru
st.sidebar.header("Masukkan Data Baru:")
island = st.sidebar.selectbox("Pulau", options=["Torgersen", "Biscoe", "Dream"])
culmen_length_mm = st.sidebar.number_input("Panjang Culmen (mm)", min_value=0.0, value=39.1, step=0.1)
culmen_depth_mm = st.sidebar.number_input("Kedalaman Culmen (mm)", min_value=0.0, value=18.7, step=0.1)
flipper_length_mm = st.sidebar.number_input("Panjang Sirip (mm)", min_value=0, value=181, step=1)
body_mass_g = st.sidebar.number_input("Berat Badan (g)", min_value=0, value=3750, step=1)
sex = st.sidebar.selectbox("Jenis Kelamin", options=["MALE", "FEMALE"])

# Masukkan data ke DataFrame
data_baru = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

st.subheader("Data Baru:")
st.write(data_baru)

# Tombol Prediksi
if st.button("Prediksi"):
    # Muat label encoder untuk kolom kategorikal
    label_encoders = load('../model/label_encoder.pkl')  # Pastikan file ini tersedia

    # Encode kolom kategorikal pada data baru
    for column in ['island', 'sex']:
        if column in label_encoders:
            data_baru[column] = label_encoders[column].transform(data_baru[column])

    # Muat model yang telah disimpan
    model_rf = load('../model/rf_model.joblib')

    # Lakukan prediksi dengan data baru
    y_pred_baru = model_rf.predict(data_baru)

    # Kembalikan hasil prediksi ke bentuk kategorikal (species)
    species_encoder = label_encoders['species']  # Pastikan ada encoder untuk 'species'
    y_pred_categorical = species_encoder.inverse_transform(y_pred_baru)

    # Kembalikan kolom island dan sex ke bentuk aslinya
    for column in ['island', 'sex']:
        if column in label_encoders:
            data_baru[column] = label_encoders[column].inverse_transform(data_baru[column])

    # Buat DataFrame untuk hasil prediksi
    hasil_prediksi = data_baru.copy()
    hasil_prediksi['Hasil Prediksi (species)'] = y_pred_categorical

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.write(hasil_prediksi)
