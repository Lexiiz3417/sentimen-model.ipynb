import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib

# ----------------- MUAT MODEL & VECTORIZER -----------------

# Pastikan file sentimen_model.h5 dan tfidf_vectorizer.joblib ada di folder yang sama
model = tf.keras.models.load_model('sentimen_model.h5')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Fungsi untuk membersihkan teks (harus sama persis dengan yang di Colab)
list_stopwords = stopwords.words('indonesian')
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in list_stopwords]
    return ' '.join(filtered_tokens)

# ----------------- KONFIGURASI FLASK -----------------

app = Flask(__name__)
CORS(app)

# Tentukan label sentimen dengan urutan yang benar
# Urutan ini harus cocok dengan urutan One-Hot Encoding saat training (0, 1, 2)
# Di mana 0=Netral, 1=Positif, 2=Negatif
sentimen_labels_full = ['Netral', 'Positif', 'Negatif']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    teks_input = request.json['teks']

    clean_teks = preprocess_text(teks_input)

    # Ubah teks bersih menjadi vektor TF-IDF
    teks_vector = vectorizer.transform([clean_teks])

    # Lakukan prediksi menggunakan model
    prediksi = model.predict(teks_vector)

    # Ambil hasil prediksi aslinya (index 0, 1, atau 2)
    prediksi_class_index = prediksi.argmax(axis=1)[0]
    prediksi_label = sentimen_labels_full[prediksi_class_index]
    
    # Konversi ke dua label (Positif atau Negatif)
    if prediksi_label == 'Positif':
        hasil_sentimen = 'Positif'
    else:
        hasil_sentimen = 'Negatif' # Sentimen Negatif dan Netral digabungkan

    return jsonify({
        'sentimen': hasil_sentimen,
        'probabilitas': float(prediksi[0][prediksi_class_index])
    })

if __name__ == '__main__':
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
        nltk.download('stopwords')

    app.run(debug=True)