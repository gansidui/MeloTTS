from transformers import AutoTokenizer, AutoModelForMaskedLM

# 下载 tokenizer

# chinese
model_id = 'bert-base-multilingual-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# english
model_id = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# french
model_id = 'dbmdz/bert-base-french-europeana-cased'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# japanese
model_id = 'tohoku-nlp/bert-base-japanese-v3'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# korean
model_id = 'kykim/bert-kor-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# spanish
model_id = 'dccuchile/bert-base-spanish-wwm-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_id)



# 下载 model

# chinese
model_id = 'hfl/chinese-roberta-wwm-ext-large'
model = AutoModelForMaskedLM.from_pretrained(model_id)

# english
model_id = 'bert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_id)

# french
model_id = 'dbmdz/bert-base-french-europeana-cased'
model = AutoModelForMaskedLM.from_pretrained(model_id)

# japanese
model_id = 'tohoku-nlp/bert-base-japanese-v3'
model = AutoModelForMaskedLM.from_pretrained(model_id)

# spanish
model_id = 'dccuchile/bert-base-spanish-wwm-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_id)
