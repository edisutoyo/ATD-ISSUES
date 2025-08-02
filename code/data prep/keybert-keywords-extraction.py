import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string

# -------------------------------------------------------
# 1. Download NLTK data if not already present:
#    nltk.download('stopwords')
#    nltk.download('punkt')
#    nltk.download('wordnet')
#    nltk.download('omw-1.4')  
#    nltk.download('averaged_perceptron_tagger') # required for pos_tag
# -------------------------------------------------------

# Helper function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default fallback

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_with_pos(tokens):
    """
    Apply POS tagging to each token and
    lemmatize based on the corresponding tag.
    """
    pos_tags = nltk.pos_tag(tokens)
    lemma_tokens = []
    for token, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        lemma_tokens.append(lemmatizer.lemmatize(token, pos=wordnet_pos))
    return lemma_tokens

def preprocess_text(text, stopwords_set):
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove punctuation
    tokens = [t for t in tokens if t not in string.punctuation]
    # Lemmatize with POS
    tokens = lemmatize_with_pos(tokens)
    # Remove stopwords
    tokens = [t for t in tokens if t.lower() not in stopwords_set]
    # Rejoin the cleaned tokens
    return " ".join(tokens)

# -----------------------------------
# Example Implementation in a Pipeline
# -----------------------------------
file_path = 'atd_final_all_indicators_tambah_baru.csv'  # update path as needed
data = pd.read_csv(file_path)

# Convert to lowercase
data['text'] = data['text'].astype(str).str.lower()

excluded_words = {
    ""
}

additional_stopwords = {
    "hadoop", "camel", "thrift", "yarn", "ruby", "apache", "rat", 
    "impala", "chrome", "eric", "namespace", "impl", "xml", "csharp", 
    "go", "perl", "apache", "org", "jboss", "aries", "maven", "pom", 
    "java", "grpc", "eol", "elk", "doe", "wa", "ha", 'https', 'one', 'set', 'activemq', 'mvc', 'http'
}

english_stopwords = set(stopwords.words('english'))
combined_stopwords = english_stopwords.union(additional_stopwords)

# Cleaning and lemmatization process
data['cleaned_text'] = data['text'].apply(lambda x: preprocess_text(x, combined_stopwords))

# Initialize KeyBERT model
kw_model = KeyBERT('all-mpnet-base-v2')

def extract_keywords_mmr(texts, top_n=25, diversity=0.5):
    all_keywords = []
    for text in texts:
        if isinstance(text, str):
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(2, 2),
                stop_words=list(combined_stopwords),
                use_mmr=True,
                top_n=top_n,
                diversity=diversity
            )
            # Filter out excluded or unwanted words
            for kw, score in keywords:
                kw_lower = kw.lower()
                kw_words = kw_lower.split()
                if not any(word in excluded_words for word in kw_words) \
                   and not any(word in additional_stopwords for word in kw_words):
                    all_keywords.append(kw_lower)
    return all_keywords

cleaned_texts = data['cleaned_text'].tolist()
keywords = extract_keywords_mmr(cleaned_texts, top_n=25, diversity=0.5)

keyword_counts = Counter(keywords)
top_50_keywords = keyword_counts.most_common(25)

print("Top 50 Keywords:")
for keyword, count in top_50_keywords:
    print(f"{keyword}: {count}")