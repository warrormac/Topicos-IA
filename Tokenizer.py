import json
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def extract_entities(tokens, tags):
    entities = []
    entity = None

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if entity:
                entities.append(entity)
            entity = {'text': token, 'type': tag[2:]}
        elif tag.startswith('I-'):
            if entity:
                entity['text'] += ' ' + token
        else:
            if entity:
                entities.append(entity)
                entity = None

    if entity:
        entities.append(entity)

    return entities

# Load the JSON files
with open('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Topicos IA/RNN/ner_promed_train.json', 'r') as train_file:
    train_data = json.load(train_file)

with open('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Topicos IA/RNN/ner_promed_test.json', 'r') as test_file:
    test_data = json.load(test_file)

# Access the samples from the JSON data
train_samples = train_data['samples']
test_samples = test_data['samples']

# Tokenize the sentences and tags using NLTK tokenizer
train_tokenized_sentences = []
train_tokenized_tags = []
for sample in train_samples:
    tokens = sample['tokens']
    tags = sample['tags']
    train_tokenized_sentences.append(word_tokenize(' '.join(tokens)))
    train_tokenized_tags.append(tags)

test_tokenized_sentences = []
test_tokenized_tags = []
for sample in test_samples:
    tokens = sample['tokens']
    tags = sample['tags']
    test_tokenized_sentences.append(word_tokenize(' '.join(tokens)))
    test_tokenized_tags.append(tags)

# Extract entities from the tokenized sentences and tags
train_entities = []
for tokens, tags in zip(train_tokenized_sentences, train_tokenized_tags):
    entities = extract_entities(tokens, tags)
    train_entities.append(entities)

test_entities = []
for tokens, tags in zip(test_tokenized_sentences, test_tokenized_tags):
    entities = extract_entities(tokens, tags)
    test_entities.append(entities)

# Save the extracted entities to a file
output_file = 'C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Topicos IA/RNN/extracted_entities.json'

output_data = {
    'train_entities': train_entities,
    'test_entities': test_entities
}

with open(output_file, 'w') as outfile:
    json.dump(output_data, outfile)

print("Extracted entities saved to:", output_file)
