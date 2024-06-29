# Import library yang diperlukan
import numpy as np  # Digunakan untuk operasi numerik
import skfuzzy as fuzz  # Digunakan untuk operasi logika fuzzy
from skfuzzy import control as ctrl  # Digunakan untuk kontrol logika fuzzy

# Fungsi untuk merekomendasikan berdasarkan rating menggunakan Fuzzy Inference System (FIS)
def fis_recommendation(rating):
    # Mendefinisikan variabel input (rating) dan output (recommendation) dengan rentang nilai
    rating_ctrl = ctrl.Antecedent(np.arange(0, 6, 1), 'rating')
    recommendation_ctrl = ctrl.Consequent(np.arange(0, 11, 1), 'recommendation')

    # Mendefinisikan fungsi keanggotaan untuk variabel input (rating)
    rating_ctrl['low'] = fuzz.trimf(rating_ctrl.universe, [0, 0, 3])
    rating_ctrl['medium'] = fuzz.trimf(rating_ctrl.universe, [0, 3, 5])
    rating_ctrl['high'] = fuzz.trimf(rating_ctrl.universe, [3, 5, 5])

    # Mendefinisikan fungsi keanggotaan untuk variabel output (recommendation)
    recommendation_ctrl['poor'] = fuzz.trimf(recommendation_ctrl.universe, [0, 0, 5])
    recommendation_ctrl['average'] = fuzz.trimf(recommendation_ctrl.universe, [0, 5, 10])
    recommendation_ctrl['good'] = fuzz.trimf(recommendation_ctrl.universe, [5, 10, 10])

    # Mendefinisikan aturan-aturan logika fuzzy
    rule1 = ctrl.Rule(rating_ctrl['low'], recommendation_ctrl['poor'])
    rule2 = ctrl.Rule(rating_ctrl['medium'], recommendation_ctrl['average'])
    rule3 = ctrl.Rule(rating_ctrl['high'], recommendation_ctrl['good'])

    # Membuat sistem kontrol fuzzy berdasarkan aturan-aturan yang telah ditentukan
    recommendation_system = ctrl.ControlSystem([rule1, rule2, rule3])
    # Membuat simulasi dari sistem kontrol fuzzy
    recommendation_sim = ctrl.ControlSystemSimulation(recommendation_system)

    # Memberikan input nilai rating ke dalam simulasi
    recommendation_sim.input['rating'] = rating
    # Melakukan komputasi untuk mendapatkan hasil rekomendasi
    recommendation_sim.compute()

    # Mengembalikan hasil rekomendasi
    return recommendation_sim.output['recommendation']
