#PP.1.10

# Treebanks são basicamente uma base de dados sentenças que são anotadas com suas estruturas gramaticais (arvores sintaticas, exemplos do substantivo na frase etc..)

# PennTreebank é um dos mais utilizados pro ingles
import nltk
from nltk.corpus import treebank
from nltk import Tree


nltk.download('treebank')

# Carregar e exibir uma sentença do Penn Treebank
sentences = treebank.sents()
print("Exemplo de sentença no Penn Treebank:")
print(sentences)


# O outro é o English Web Treebank porém ele basicamente "tokeniza" as palavras da sentença, não sei bem se é isso que o professor quer, perguntar na apresentação.