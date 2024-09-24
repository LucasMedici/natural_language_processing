import time
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import spacy
import stanfordnlp
import nltk


nltk.download('punkt')
stanfordnlp.download('en')  # Baixar modelo para inglês
nlp_spacy = spacy.load("en_core_web_sm")  # Modelo para inglês
nlp_stanford = stanfordnlp.Pipeline(processors='tokenize', lang='en')


def scrape_reviews(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extraindo os textos das reviews
    reviews = soup.find_all('div', class_='text show-more__control')
    review_texts = [review.get_text(strip=True) for review in reviews if review.get_text(strip=True)]
    return review_texts

# Tokenizador usando NLTK
def nltk_tokenizer(text):
    return sent_tokenize(text, language='english')

# Tokenizador usando SpaCy
def spacy_tokenizer(text):
    doc = nlp_spacy(text)
    return [sent.text for sent in doc.sents]

# Tokenizador usando StanfordNLP
def stanford_tokenizer(text):
    doc = nlp_stanford(text)
    return [" ".join([token.text for token in sentence.tokens]) for sentence in doc.sentences]



def measure_performance(tokenizer_func, reviews, tokenizer_name):
    start_time = time.time()
    total_tokens = 0
    tokenized_output = []

    for review in reviews:
        tokens = tokenizer_func(review)
        total_tokens += len(tokens)
        tokenized_output.append(tokens)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Desempenho do {tokenizer_name}:")
    print(f"Total de tokens: {total_tokens}")
    print(f"Tempo de execução: {execution_time:.4f} segundos\n")
    
    return total_tokens, execution_time, tokenized_output

if __name__ == "__main__":
    url = "https://www.imdb.com/title/tt1196946/reviews/"
    reviews = scrape_reviews(url)

    # Comparar desempenho de cada tokenizador
    nltk_result, nltk_execution_time, nltk_tokens = measure_performance(nltk_tokenizer, reviews, "NLTK")
    spacy_result, spacy_execution_time, spacy_tokens = measure_performance(spacy_tokenizer, reviews, "SpaCy")
    stanford_result, stanford_execution_time, stanford_tokens = measure_performance(stanford_tokenizer, reviews, "StanfordNLP")
    

    print("\nSaídas de tokenização:")
    print(f"\nReview 4 de exemplo:")
    print(f"Texto Original: {reviews[4]}")
    print(f"NLTK Tokens: {nltk_tokens[4]}")
    print(f"SpaCy Tokens: {spacy_tokens[4]}")
    print(f"StanfordNLP Tokens: {stanford_tokens[4]}")
