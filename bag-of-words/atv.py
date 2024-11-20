from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk

# Baixando NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords


# 3 exemplos de reviews de um celular para servir pro teste
reviews = [
    "This phone is great, the camera quality is amazing and it lasts all day!",
    "The phone has a decent camera, but the battery doesn't last long at all.",
    "I absolutely love this camera, it takes beautiful photos, but the phone battery is poor."
]

# pré processamento, deixando texto minusculo, tirando pontuação e stopwords (preposições e zaes)
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# rodando o pré processamento em todas as 3 reviews
processed_reviews = [preprocess(review) for review in reviews]


# Roda TF-IDF Vectorizer (técnica de frequencia e importancia de palavras em uma frase)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_reviews)
print("Matriz TF-IDF:\n", tfidf_matrix.toarray())


# Calculo de similaridade com a função cosseno
cosine_sim = cosine_similarity(tfidf_matrix)
print("\nMatriz de Similaridade de Cosseno:\n", cosine_sim)


# Printando o valor da similaridade entre cada review
for i in range(len(reviews)):
    for j in range(i + 1, len(reviews)):
        print(f"Similaridade entre Review {i + 1} e Review {j + 1}: {cosine_sim[i][j]}")