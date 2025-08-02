import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from fuzzywuzzy import fuzz

# Load SpaCy model for POS tagging and lemmatization
nlp = spacy.load("en_core_web_trf")

# Load a pre-trained SentenceTransformer model for embeddings
model = SentenceTransformer('all-mpnet-base-v2')
kw_model = KeyBERT(model=model)

# Parameters for the ATD keyword extraction
n_iterations = 5  # Number of iterations
top_k = 10         # Number of top keywords to keep per document
batch_size = 5     # Process documents in small batches
number_newseed = 3 # Number of new seed keywords per iteration

# Custom list of stop words for ATD
custom_stop_words = [
    "netty", "function", "process", "data", "service", "jetty", "camel", 
    "hbase", "thrift", "jena", "httpaction", "bookkeeper", "shards", "junit",
    "linux", "ubuntu", "hadoop", "fuseki", "nifi", "flink", "yarn", "abdera", "jdk", "aries", "airflow",
    "rat", "apache", "mapred", "eol", "python", "jboss"
]

# Check if a phrase contains only repeated tokens
def is_repetitive_phrase(phrase):
    cleaned = phrase.replace('"', '').strip()
    tokens = cleaned.split()
    return len(set(tokens)) == 1

# Preprocess text: remove stopwords, lemmatize, and filter POS tags
def preprocess_text(text, stop_words, pos_tags_to_keep=None):
    doc = nlp(text.lower())
    cleaned_words = [
        token.lemma_ for token in doc
        if token.text not in stop_words
        and token.pos_ in pos_tags_to_keep
        and not token.is_stop
        and token.is_alpha
    ]
    return ' '.join(cleaned_words)

# Check if a new keyword is similar to any in the existing set
def is_similar_keyword(keyword, keyword_set):
    return any(fuzz.ratio(keyword, kw) > 85 for kw in keyword_set)

# Normalize lemmatized keywords (optional)
def normalize_keywords(keywords):
    normalized = {}
    for word in keywords:
        lemma = ' '.join([token.lemma_ for token in nlp(word)])
        normalized[lemma] = word
    return list(normalized.keys())

# ATD-specific keyword extraction using iterative expansion
def atd_specific_keyword_extraction(documents, seed_keywords):
    final_keywords = set(seed_keywords)
    keyword_scores = {}
    pos_tags_to_keep = {"NOUN", "VERB", "ADJ"}

    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        scored_candidates = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            for doc in batch:
                cleaned_doc = preprocess_text(doc, custom_stop_words, pos_tags_to_keep)

                keywords_candidates = kw_model.extract_keywords(
                    cleaned_doc,
                    keyphrase_ngram_range=(1, 1),
                    stop_words='english',
                    use_maxsum=False,
                    use_mmr=True,
                    diversity=0.5,
                    top_n=top_k,
                    seed_keywords=list(final_keywords)
                )

                for keyword, _ in keywords_candidates:
                    if is_similar_keyword(keyword, final_keywords):
                        continue

                    keyword_embedding = model.encode([keyword], convert_to_tensor=False)
                    seed_embeddings = model.encode(list(final_keywords), convert_to_tensor=False)

                    avg_score = np.mean(cosine_similarity(keyword_embedding, seed_embeddings).flatten())
                    max_score = np.max(cosine_similarity(keyword_embedding, seed_embeddings).flatten())
                    final_score = (avg_score + max_score) / 2
                    scored_candidates.append((keyword, final_score))

        scored_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

        top_new_seed_candidates = []
        for kw, score in scored_candidates:
            if kw not in final_keywords:
                top_new_seed_candidates.append((kw, score))
            if len(top_new_seed_candidates) >= number_newseed:
                break

        print("Added keywords in this iteration:")
        for kw, score in top_new_seed_candidates:
            final_keywords.add(kw)
            keyword_scores[kw] = score
            print(f"  {kw} -> {score:.4f}")

    sorted_keyword_scores = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keyword_scores

# Main function to run the keyword extraction pipeline
def main():
    csv_path = "atd_final_all_indicators.csv"  # Update as needed
    df = pd.read_csv(csv_path)

    if 'text' not in df.columns:
        print("Error: CSV file must have a 'text' column")
        return

    documents = df['text'].fillna("").tolist()
    seed_keywords = ["move", "refactor", "remove", "dependency", "couple", "update"]

    final_keywords_with_scores = atd_specific_keyword_extraction(documents, seed_keywords)

    print("\nFinal ATD-Specific Keywords with Scores (sorted by score):")
    for keyword, score in final_keywords_with_scores:
        print(f"{keyword}: {score:.4f}")

if __name__ == "__main__":
    main()
