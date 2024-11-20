from transformers import T5Tokenizer, T5ForConditionalGeneration

# Carregar o modelo e o tokenizer
modelo = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(modelo)
model = T5ForConditionalGeneration.from_pretrained(modelo)

# Texto para sumarização
texto = """
O Processamento de Linguagem Natural (PLN) é uma área da Inteligência Artificial que tem como objetivo permitir que as máquinas compreendam, interpretem e gerem a linguagem humana de forma útil. A sumarização automática é uma das tarefas mais importantes do PLN, e pode ser dividida em duas abordagens principais: a sumarização extrativa e a sumarização abstrativa. A sumarização extrativa seleciona partes do texto original, enquanto a abstrativa gera novos trechos de texto que representam a ideia central do conteúdo.
"""

# Pré-processamento do texto
inputs = tokenizer("summarize: " + texto, return_tensors="pt", max_length=512, truncation=True)

# Gerar a sumarização
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decodificar e exibir o resumo gerado
resumo = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Resumo:", resumo)
