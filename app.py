# Impor pustaka yang diperlukan dari Flask dan modul rekomendasi lainnya
from flask import Flask, request, render_template
import pandas as pd
from fis import fis_recommendation
from neural_network import neural_network_recommendation
from neuro_fuzzy import neuro_fuzzy_recommendation
from svm_genetic import svm_genetic_recommendation

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Baca dataset buku ke dalam DataFrame
books_df = pd.read_csv('books.csv')

# Rute untuk halaman utama
@app.route('/')
def index():
    # Render halaman index.html dengan data buku dari DataFrame
    return render_template('index.html', books=books_df.to_dict(orient='records'))

# Rute untuk menangani permintaan rekomendasi
@app.route('/recommend', methods=['POST'])
def recommend():
    # Ambil user_id dan book_id dari form yang dikirim
    user_id = int(request.form['user_id'])
    book_id = int(request.form['book_id'])
    
    # Cari rating buku berdasarkan user_id dan book_id
    rating = books_df[(books_df['user_id'] == user_id) & (books_df['book_id'] == book_id)]['rating'].values[0]

    # Dapatkan skor rekomendasi menggunakan berbagai metode
    fis_score = fis_recommendation(rating)
    nn_score = neural_network_recommendation(book_id, rating)
    nf_score = neuro_fuzzy_recommendation(book_id, rating)
    svm_gen_score = svm_genetic_recommendation(book_id, rating)

    # Ambil informasi buku berdasarkan book_id
    book_info = books_df[books_df['book_id'] == book_id].iloc[0]

    # Penjelasan rekomendasi berdasarkan skor dari masing-masing metode
    fis_recommendation_str = explain_recommendation(fis_score)
    nn_recommendation_str = explain_recommendation(nn_score)
    nf_recommendation_str = explain_recommendation(nf_score)
    svm_gen_recommendation_str = explain_recommendation(svm_gen_score)

    # Render halaman result.html dengan informasi buku, skor rekomendasi, dan penjelasan
    return render_template('result.html', book_info=book_info, fis_score=fis_score, nn_score=nn_score, nf_score=nf_score,
                           svm_gen_score=svm_gen_score, fis_recommendation=fis_recommendation_str,
                           nn_recommendation=nn_recommendation_str, nf_recommendation=nf_recommendation_str,
                           svm_gen_recommendation=svm_gen_recommendation_str)

# Fungsi untuk memberikan penjelasan rekomendasi berdasarkan skor
def explain_recommendation(score):
    if score < 3:
        return "Kurang direkomendasikan"
    elif score < 7:
        return "Direkomendasikan dengan baik"
    else:
        return "Sangat direkomendasikan"

# Jalankan aplikasi Flask dalam mode debug
if __name__ == '__main__':
    app.run(debug=True)
