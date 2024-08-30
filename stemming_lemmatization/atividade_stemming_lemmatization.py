#PP.1.2

from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import ToktokTokenizer

import spacy
from nltk.tokenize import word_tokenize
import stanza


# Stemming (partes em comum das palavras)

tokenizer = ToktokTokenizer() # Biblioteca pra dividir as frases
snowball_stemmer = SnowballStemmer(language='portuguese') # Stemer PT BR

texto1 = "Os meninos correram rapidamente para a escola."
texto2 = "O menino estava correndo muito rápido."

tokens_texto1 = tokenizer.tokenize(texto1)
tokens_texto2 = tokenizer.tokenize(texto2)

stems_texto1 = [snowball_stemmer.stem(word) for word in tokens_texto1]
stems_texto2 = [snowball_stemmer.stem(word) for word in tokens_texto2]

print("-------------")
print("Steamming dos textos")
print(stems_texto1)
print(stems_texto2)
print('-------------')



# Lemmatization (sentido geral da palavra, correndo = correr por exemplo)
spacy_nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])

lemmas_texto1 = [token.lemma_ for token in spacy_nlp(texto1)]
lemmas_texto2 = [token.lemma_ for token in spacy_nlp(texto2)]

print("-------------")
print("Lemmatization dos textos")
print(lemmas_texto1)
print(lemmas_texto2)
print('-------------')

