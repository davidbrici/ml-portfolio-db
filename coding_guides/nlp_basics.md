# Basic NLP Guide for ML Engineers

## Introduction to NLP
**Natural Language Processing (NLP)** is a subfield of AI that enables machines to understand, interpret, and generate human language. NLP is widely used for **chatbots, sentiment analysis, text classification, translation, and more**.

---
## 1. Setting Up the Environment
### Installing Required Libraries
```bash
pip install nltk spacy transformers scikit-learn pandas
```

### Importing Libraries
```python
import nltk
import spacy
import transformers
import pandas as pd
```

---
## 2. Tokenization
Tokenization is the process of splitting text into words or sentences.

### Using NLTK
```python
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
text = "Machine Learning is amazing! NLP is a part of AI."
print(word_tokenize(text))  # Word Tokenization
print(sent_tokenize(text))  # Sentence Tokenization
```

### Using spaCy
```python
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print([token.text for token in doc])
```

---
## 3. Removing Stopwords
```python
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
filtered_text = [word for word in word_tokenize(text) if word.lower() not in stop_words]
print(filtered_text)
```

---
## 4. Stemming and Lemmatization
### Stemming with NLTK
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "flies", "easily", "fairly"]
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)
```

### Lemmatization with spaCy
```python
doc = nlp("running flies easily fairly")
print([token.lemma_ for token in doc])
```

---
## 5. Named Entity Recognition (NER)
```python
text = "Apple was founded by Steve Jobs in California."
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
```

---
## 6. Text Vectorization
### TF-IDF Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["This is a sentence", "This is another sentence"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
```

---
## 7. Sentiment Analysis
### Using VADER (NLTK)
```python
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("I love NLP!"))
```

---
## 8. Text Classification with Machine Learning
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = ["I love this movie", "This is a bad movie"]
labels = [1, 0]  # 1 = Positive, 0 = Negative

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)
print(model.predict(["This movie is amazing"]))
```

---
## 9. Transformer Models for NLP
### Using Hugging Face Transformers for Text Generation
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
print(generator("Machine learning is", max_length=30))
```

### Using BERT for Text Classification
```python
classifier = pipeline("sentiment-analysis")
print(classifier("I love machine learning!"))
```

---
## 10. Named Entity Recognition with Transformers
```python
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
print(ner("Elon Musk founded Tesla in California."))
```

---
## Conclusion
NLP is a **powerful field** that enables machines to **process and generate human language**. Understanding **tokenization, vectorization, sentiment analysis, and transformers** is crucial for ML engineers.

For more advanced topics, check out **LSTMs, Transformers, and NLP Fine-tuning**!

Happy coding! 🚀
