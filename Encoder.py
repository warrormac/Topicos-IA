import numpy as np
import tensorflow as tf
import json

# Load the cleaned entities
with open('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Topicos IA/RNN/cleaned_entities.json', 'r') as entities_file:
    entities_data = json.load(entities_file)

train_entities = entities_data['train_entities']
test_entities = entities_data['test_entities']

# Initialize tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()

# Collect all texts for fitting tokenizer
all_texts = []
for entity_group in train_entities:
    for entity in entity_group:
        all_texts.append(entity['text'])
for entity_group in test_entities:
    for entity in entity_group:
        all_texts.append(entity['text'])

# Fit tokenizer on all texts
tokenizer.fit_on_texts(all_texts)

# Convert entities to encoded sequences and labels
encoded_sequences = []
labels = []
label2idx = {'Disease': 0, 'Location': 1, 'Number_of_cases': 2, 'Date': 3}  # Mapping of labels to numerical form

for entity_group in train_entities:
    for entity in entity_group:
        encoded_sequences.append(tokenizer.texts_to_sequences([entity['text']])[0])
        labels.append(label2idx.get(entity['type'], label2idx['Date']))

for entity_group in test_entities:
    for entity in entity_group:
        encoded_sequences.append(tokenizer.texts_to_sequences([entity['text']])[0])
        labels.append(label2idx.get(entity['type'], label2idx['Date']))

# Pad sequences to have the same length
max_sequence_length = max(len(seq) for seq in encoded_sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(encoded_sequences, maxlen=max_sequence_length)

# Convert padded sequences and labels to tensors
padded_sequences_tensor = tf.convert_to_tensor(padded_sequences)
labels_tensor = tf.one_hot(labels, len(label2idx))

# Print the tensor shapes for verification
print("Padded sequences tensor shape:", padded_sequences_tensor.shape)
print("Labels tensor shape:", labels_tensor.shape)

# Save padded sequences tensor
np.save('padded_sequences.npy', padded_sequences_tensor)

# Save labels tensor
np.save('labels.npy', labels_tensor)
