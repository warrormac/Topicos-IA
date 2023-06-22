import json
import re

# Load the JSON file
with open('C:/Users/Andre/OneDrive/UCSP/Semestre 7 Online/Topicos IA/RNN/extracted_entities.json', 'r') as file:
    data = json.load(file)

# Extract entities from the "train_entities" section
train_entities = data['train_entities']
cleaned_train_entities = []

# Clean each entity text
for entity_list in train_entities:
    cleaned_entities = []
    for entity in entity_list:
        cleaned_text = re.sub(r'[^\w\s]', '', entity['text'])  # Remove special characters
        cleaned_text = cleaned_text.lower()  # Convert text to lowercase
        cleaned_entities.append({'text': cleaned_text, 'type': entity['type']})
    cleaned_train_entities.append(cleaned_entities)

# Extract entities from the "test_entities" section
test_entities = data['test_entities']
cleaned_test_entities = []

# Clean each entity text
for entity_list in test_entities:
    cleaned_entities = []
    for entity in entity_list:
        cleaned_text = re.sub(r'[^\w\s]', '', entity['text'])  # Remove special characters
        cleaned_text = cleaned_text.lower()  # Convert text to lowercase
        cleaned_entities.append({'text': cleaned_text, 'type': entity['type']})
    cleaned_test_entities.append(cleaned_entities)

# Save the cleaned entities to a new JSON file
cleaned_data = {'train_entities': cleaned_train_entities, 'test_entities': cleaned_test_entities}
with open('cleaned_entities.json', 'w') as file:
    json.dump(cleaned_data, file)

print("Text cleaning completed. Cleaned entities saved to 'cleaned_entities.json'.")
