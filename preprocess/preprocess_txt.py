import re


filepath = "data/shakespeare.txt"
pattern = r'[^a-zA-Z\s]'
with open(filepath, 'r') as file:
    text = file.read()
clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
with open('data/shakespeare.txt', 'w') as file:
    file.write(clean_text)