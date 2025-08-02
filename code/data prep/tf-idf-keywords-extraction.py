import pandas as pd
import re  
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure consistent language detection results
DetectorFactory.seed = 0

# ----------------------------
# Configuration
# ----------------------------
CSV_FILE_PATH = "atd_final_all_indicators.csv"  # Replace with your actual CSV file path
TEXT_COLUMN = 'text'                       # Name of the text column
TOP_N = 25                                 # Number of top keywords
PROJECT_NAMES = ['camel', 'thrift', 'hbase', 'hadoop', 'apache', 'ws0432', 'rat', 'itd', 'ct', 'airflow4535', 'jobspy',
                 'airflow', 'junit', 'smx', 'debian', 'conf', 'perl', 'python', 'initnewcluster', 'localtaskexecutor',
                 'ubuntu', 'sitepackages', 'distutils', 'josh', 'jdk', 'jasypt', 'libs', 'jbang', 'mainhelpersetpropertiesontargetmainhelperjava', 'hadooptools',
                 'cameljgropus', 'camelsql', 'concurrentmodificationexception', 'shellprofiles', 'bytessource', 'assertj', 'httpsgithubcombelabanjgroupsdiscussionsdiscussioncomment', 'oahhclient', 'bean',
                 'mqtt', 'avrohttpsgithubcomapacheavroblobreleaselangjavaavrosrcmainjavaorgapacheavroschemajavall', 'camelatom', 'yarn', 'google', 'abdera', 'cfx', 'camelcxf', 'yang', 'id', 'camelgrpc', 'net', 'namevalidator']  # Additional project names as stopwords
LANGUAGE_FILTER = 'en'  # Filter for English language only
# ----------------------------

# Load the dataset
df = pd.read_csv(CSV_FILE_PATH)

# Preprocessing setup
stop_words = set(stopwords.words('english')) | set(PROJECT_NAMES)  # Add project names to stopwords
lemmatizer = WordNetLemmatizer()

# Mapping NLTK POS tags to WordNet POS tags for lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

# Language filtering function
def is_english(text):
    try:
        return detect(text) == LANGUAGE_FILTER
    except LangDetectException:
        return False

# Preprocessing function with language filter
def preprocess(text):
    if not is_english(text):
        return ""  
    
    # Lowercase
    text = text.lower()
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7f]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenisasi dan POS tagging
    tokens = nltk.word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Lemmatization dan stopword removal
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in stop_words
    ]
    return " ".join(tokens)

# Apply preprocessing to the text column
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(preprocess)

# Drop rows that became empty after preprocessing
df = df[df[TEXT_COLUMN] != ""]

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=1000,       # Adjust as needed
    ngram_range=(3, 3),      # Use trigrams
    stop_words='english'
)

# Fit-transform the text data
tfidf_matrix = tfidf.fit_transform(df[TEXT_COLUMN])

# Extract feature names
feature_names = tfidf.get_feature_names_out()

# Compute average TF-IDF score for each term
avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1  # Convert to 1D array

# Map terms to their scores
feature_scores = list(zip(feature_names, avg_tfidf_scores))
feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)

# Print top N keywords
print("Top {} Keywords:".format(TOP_N))
for keyword, score in feature_scores[:TOP_N]:
    print(f"{keyword}: {score:.4f}")
