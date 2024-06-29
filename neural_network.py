# neural_network.py

# Impor pustaka yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Definisikan fungsi untuk rekomendasi buku berbasis jaringan saraf
def neural_network_recommendation(book_id, rating):
    # Baca dataset yang berisi informasi buku
    df = pd.read_csv('books.csv')
    
    # Pilih fitur dan variabel target dari dataset
    X = df[['book_id', 'rating']]
    y = df['rating']
    
    # Bagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi Multi-Layer Perceptron Regressor dengan parameter yang ditentukan
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000)
    
    # Latih model menggunakan data pelatihan
    model.fit(X_train, y_train)
    
    # Prediksi rating untuk book_id dan rating yang diberikan
    prediction = model.predict([[book_id, rating]])
    
    # Kembalikan rating yang diprediksi
    return prediction[0]
