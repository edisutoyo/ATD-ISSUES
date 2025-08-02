import pandas as pd
import spacy
import string
import re
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------
# Initialize spaCy English model
# ---------------------------------------
nlp = spacy.load("en_core_web_trf")

# Ensure "move" is removed from spaCy stopwords
if "move" in nlp.Defaults.stop_words:
    nlp.Defaults.stop_words.remove("move")

# Force reinitialize stopwords for the current pipeline
nlp.vocab["move"].is_stop = False

# ---------------------------------------
# Initialize Sentence-BERT model
# ---------------------------------------
model_name = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(model_name)

# ---------------------------------------
# List of keywords
# ---------------------------------------
keywords = ['move', 'could', 'test', 'version', 'package', 'dependency', 'upgrade', 'class', 'use', 'also', 'need', 'like', 'api', 'build', 'library']

# N-gram setting
n_gram = 1

excluded_words = {}

# ---------------------------------------
# 1. Preprocessing Function (spaCy)
#    + Removes links/URLs & code snippets (backticks)
# ---------------------------------------
def preprocess_text(text):
    """
    Preprocess text by:
      - Removing CamelCase words
      - Removing links/websites (regex)
      - Removing inline/triple backtick code snippets
      - Lowercasing text
      - Tokenizing & POS tagging via spaCy
      - Removing stopwords, punctuation, whitespace
      - Keeping only certain POS tags (NOUN, VERB, ADJ, ADV, AUX, PART)
      - Lemmatization
    Returns preprocessed string.
    """
    text = re.sub(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', '', text)  # Remove CamelCase
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove triple backtick code
    text = re.sub(r'`[^`]*`', '', text)  # Remove single backtick code
    text = text.lower()

    doc = nlp(text)
    clean_tokens = []
    for token in doc:
        if token.is_stop and token.text != "move":
            continue
        if token.is_punct or token.is_space:
            continue
        if token.pos_ not in ["NOUN", "VERB", "ADJ", "ADV", "AUX", "PART"]:
            continue
        lemma = token.lemma_.strip()
        if lemma in excluded_words or lemma in string.punctuation or lemma == "":
            continue
        clean_tokens.append(lemma)
    return " ".join(clean_tokens)

# ---------------------------------------
# 2. N-Gram Generator
# ---------------------------------------
def generate_n_grams(text, n):
    """
    Convert sentence (text) into n-grams based on the given n value.
    """
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

# ---------------------------------------
# 3. Main Function for CSV Processing
# ---------------------------------------
def process_csv(input_csv: str, output_csv: str, threshold: float = 0.9):
    """
    Reads CSV (delimiter=';'), combines Summary and Description columns,
    applies preprocessing + chunking + similarity computation,
    then writes the output to output_csv.

    The 'threshold' parameter determines "ATD-Related" vs "Non-ATD" labels.
    """
    df = pd.read_csv(input_csv, delimiter=';')
    df["Summary_Description"] = (df["Summary"].fillna("") + " - " + df["Description"].fillna(""))

    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)
    processed_rows = []

    for idx, row in df.iterrows():
        issue_id = row.get("Key", f"unknown-{idx}")
        original_text = str(row["Summary_Description"])
        preprocessed = preprocess_text(original_text)
        chunks = generate_n_grams(preprocessed, n_gram)

        best_keyword = ""
        best_similarity_score = 0.0
        label = "Non-ATD"

        if chunks:
            chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
            similarities = util.cos_sim(chunk_embeddings, keyword_embeddings)
            import torch
            max_sim_values, max_sim_indices = torch.max(similarities, dim=1)
            overall_best_sim_value = torch.max(max_sim_values)
            best_chunk_idx = torch.argmax(max_sim_values).item()
            best_chunk_similarity = overall_best_sim_value.item()
            best_keyword_idx = max_sim_indices[best_chunk_idx].item()

            best_keyword = keywords[best_keyword_idx]
            best_similarity_score = best_chunk_similarity

            if best_chunk_similarity > threshold:
                label = "ATD-Related"

        processed_rows.append({
            "Key": issue_id,
            "Summary_Description_Original": original_text,
            "Summary_Description_Preprocessed": preprocessed,
            "closest_keyword": best_keyword,
            "similarity_score": best_similarity_score,
            "label": label
        })

    output_df = pd.DataFrame(processed_rows)
    output_df.to_csv(output_csv, index=False, sep=';')
    print(f"Processing result has been saved to '{output_csv}'.")

# ---------------------------------------
# 4. Process Multiple CSV Files
# ---------------------------------------
if __name__ == "__main__":
    input_files = [
        "../APACHE_CAMEL.csv",
        "../APACHE_NETBEANS.csv",
        "../APACHE_ACTIVEMQ.csv",
        "../APACHE_CASSANDRA.csv",
        "../APACHE_DRILL.csv",
        "../APACHE_GEODE.csv",
        "../APACHE_KAFKA.csv",
        "../APACHE_LUCENE.csv",
        "../APACHE_SOLR.csv",
        "../APACHE_SPARK.csv"
    ]

    output_files = [
        "APACHE_CAMEL-bigram.csv",
        "APACHE_NETBEANS-bigram.csv",
        "APACHE_ACTIVEMQ-bigram.csv",
        "APACHE_CASSANDRA-bigram.csv",
        "APACHE_DRILL-bigram.csv",
        "APACHE_GEODE-bigram.csv",
        "APACHE_KAFKA-bigram.csv",
        "APACHE_LUCENE-bigram.csv",
        "APACHE_SOLR-bigram.csv",
        "APACHE_SPARK-bigram.csv"
    ]

    for in_file, out_file in zip(input_files, output_files):
        process_csv(in_file, out_file, threshold=0.9)

    print("Processing completed for all files!")