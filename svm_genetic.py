# svm_genetic.py

# Impor pustaka yang diperlukan
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from deap import base, creator, tools, algorithms

# Definisikan fungsi rekomendasi menggunakan SVM dan algoritma genetik
def svm_genetic_recommendation(book_id, rating):
    # Baca dataset buku ke dalam DataFrame
    df = pd.read_csv('books.csv')
    
    # Pilih fitur dan target dari dataset
    X = df[['book_id', 'rating']]
    y = df['rating']
    
    # Inisialisasi model SVM Regressor
    model = SVR()
    
    # Latih model menggunakan data
    model.fit(X, y)
    
    # Definisikan fungsi evaluasi untuk algoritma genetik
    def eval(individual):
        book_id, rating = individual
        prediction = model.predict([[book_id, rating]])
        return prediction[0],
    
    # Buat tipe kebugaran dan individu untuk algoritma genetik
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Buat toolbox untuk mendefinisikan operator genetik
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", eval)
    
    # Inisialisasi populasi dengan 10 individu
    population = toolbox.population(n=10)
    
    # Jalankan algoritma genetik untuk mengembangkan populasi
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)
    
    # Pilih individu terbaik dari populasi
    best_ind = tools.selBest(population, 1)[0]
    
    # Kembalikan nilai kebugaran individu terbaik
    return best_ind.fitness.values[0]
