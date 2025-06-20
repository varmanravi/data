import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from your_script_executor import execute_script  # Your module to run DB scripts

# Step 1: Read data
df = pd.read_excel("ritm_requests.xlsx")
descriptions = df['description'].fillna("")

# Step 2: Basic preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.strip()

df['cleaned'] = descriptions.map(clean_text)

# Step 3: Train or Load intent classifier
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])  # or load saved model
# y = ... labels if available
# classifier = LogisticRegression().fit(X, y)

# Step 4: Inference Example
# intent = classifier.predict(vectorizer.transform([clean_text_input]))[0]

# Step 5: Named Entity Recognition for extracting parameters
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Step 6: Example execution
# if intent == "check_db_size":
#     db_name = extract_entities(cleaned_text)[0][0]
#     execute_script("check_db_size.sh", db=db_name)
