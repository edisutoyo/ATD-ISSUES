import pandas as pd
import spacy
import string
import re
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------
# Initialize spaCy English model
# ---------------------------------------
nlp = spacy.load("en_core_web_trf")

# Ensure "move" is not treated as a stopword
if "move" in nlp.Defaults.stop_words:
    nlp.Defaults.stop_words.remove("move")

nlp.vocab["move"].is_stop = False

# ---------------------------------------
# Initialize Sentence-BERT model
# ---------------------------------------
model_name = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(model_name)

# ---------------------------------------
# Define keywords to use for similarity comparison
# ---------------------------------------
keywords = ['use', 'test', 'package', 'upgrade', 'file', 'version', 'need', 'like', 'class', 'dependency', 'remove', 'place', 'old', 'add', 'new']

# N-gram setting
n_gram = 1

excluded_words = {}

# ---------------------------------------
# 1. Preprocessing function using spaCy
#    + Removes URLs & code snippets
# ---------------------------------------
def preprocess_text(text):
    """
    Preprocess text by:
      - Removing CamelCase words
      - Removing URLs
      - Removing code snippets within backticks
      - Lowercasing
      - Tokenizing with spaCy
      - Removing stopwords, punctuation, whitespace
      - Keeping only specific POS tags (NOUN, VERB, ADJ, ADV, AUX, PART)
      - Lemmatizing
    Returns cleaned text.
    """
    text = re.sub(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', '', text)
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
        if lemma in excluded_words:
            continue
        if lemma in string.punctuation or lemma == "":
            continue

        clean_tokens.append(lemma)

    return " ".join(clean_tokens)

# ---------------------------------------
# 2. Function to create n-grams
# ---------------------------------------
def generate_n_grams(text, n):
    """
    Converts text into a list of n-grams.
    """
    words = text.split()
    length = len(words)
    return [" ".join(words[i:i+n]) for i in range(length - n + 1)]

# ---------------------------------------
# 3. Main function to process a CSV file
# ---------------------------------------
def process_csv(input_csv: str, output_csv: str, threshold: float = 0.9):
    """
    Processes a CSV file (delimiter=';'), combines Summary and Description,
    performs preprocessing, chunking, and similarity computation,
    then writes the results to an output CSV.

    'threshold' is used to classify "ATD-Related" vs "Non-ATD".
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

        if len(chunks) > 0:
            chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
            similarities = util.cos_sim(chunk_embeddings, keyword_embeddings)

            import torch
            max_sim_values, max_sim_indices = torch.max(similarities, dim=1)
            overall_best_sim_value = torch.max(max_sim_values)
            overall_best_sim_idx = torch.argmax(max_sim_values)

            best_chunk_idx = overall_best_sim_idx.item()
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
    print(f"Processing results saved to '{output_csv}'.")

# ---------------------------------------
# 4. Run batch processing for multiple CSV files
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
        "APACHE_CAMEL-unigram.csv",
        "APACHE_NETBEANS-unigram.csv",
        "APACHE_ACTIVEMQ-unigram.csv",
        "APACHE_CASSANDRA-unigram.csv",
        "APACHE_DRILL-unigram.csv",
        "APACHE_GEODE-unigram.csv",
        "APACHE_KAFKA-unigram.csv",
        "APACHE_LUCENE-unigram.csv",
        "APACHE_SOLR-unigram.csv",
        "APACHE_SPARK-unigram.csv"
    ]

    for in_file, out_file in zip(input_files, output_files):
        process_csv(in_file, out_file, threshold=0.9)

    print("Processing completed for all files!")