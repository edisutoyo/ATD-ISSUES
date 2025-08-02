import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split
from small_text.query_strategies import LeastConfidence, BreakingTies, RandomSampling
from classification.active_learning import fullpipeline
from nltk.tokenize import sent_tokenize

# === Load original CSV ===
csv_file = "dataset.csv"
df = pd.read_csv(csv_file)

# === Gabungkan Summary dan Description jika ada ===
if "Summary" in df.columns and "Description" in df.columns:
    df["Summary_Description"] = df["Summary"].fillna("") + ". " + df["Description"].fillna("")
elif "Summary_Description" not in df.columns:
    raise ValueError("Kolom Summary_Description tidak ditemukan dan Summary/Description tidak tersedia.")



# def clean_jira_text_v2(text):
#     text = str(text).lower()

#     # 1. Hapus {code}, {noformat}, {color}, dsb.
#     text = re.sub(r'\{code(:[a-z]+)?\}.*?\{code\}', ' ', text, flags=re.DOTALL)
#     text = re.sub(r'\{noformat\}.*?\{noformat\}', ' ', text, flags=re.DOTALL)
#     text = re.sub(r'\{color:[^}]+\}', '', text)
#     text = re.sub(r'\n+', ' ', text)

#     # 2. Hapus blok diff (gunakan pola diff dan "< --- >" atau "---")
#     text = re.sub(r'(?m)^.*diff.*$', '', text)
#     text = re.sub(r'(?m)^<.*---.*$', '', text)
#     text = re.sub(r'(?m)^>.*$', '', text)
#     text = re.sub(r'(?m)^\d+c\d+$', '', text)  # pola baris patch/diff

#     # 3. Hapus baris yang merupakan kode Java/kelas/import (lebih dari 2 baris berturut-turut kode)
#     lines = text.split('\n')
#     clean_lines = []
#     code_buffer = []
#     for line in lines:
#         # Deteksi baris kode (berisi '{', '}', 'class', 'import', ';', dsb.)
#         if re.match(r'^\s*(import |package |public |private |protected |class |@|/|\*|\{|}|;)', line.strip()):
#             code_buffer.append(line)
#         else:
#             if len(code_buffer) >= 2:
#                 # Hapus code block panjang
#                 code_buffer = []
#             else:
#                 clean_lines.extend(code_buffer)
#                 code_buffer = []
#             clean_lines.append(line)
#     # Sisa code_buffer kecil dimasukkan
#     if len(code_buffer) < 2:
#         clean_lines.extend(code_buffer)
#     # Gabungkan lagi dan bersihkan whitespace
#     text = ' '.join([l.strip() for l in clean_lines if l.strip() != ''])
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # Contoh penggunaan:
# df["Summary_Description"] = df["Summary_Description"].apply(clean_jira_text_v2)


def clean_jira_text(text):
    # Lowercase
    text = str(text).lower()
    
    # Hapus block {code:...}{code}
    text = re.sub(r'\{code(:[a-z]+)?\}.*?\{code\}', ' ', text, flags=re.DOTALL)
    # Hapus block {noformat}...{noformat}
    text = re.sub(r'\{noformat\}.*?\{noformat\}', ' ', text, flags=re.DOTALL)
    # Hapus tag warna/markup Jira lain
    text = re.sub(r'\{color:[^}]+\}', '', text)
    # Hapus baris yang terlalu teknis/metadata
    # text = re.sub(r'(?m)^\s*(plan after changes looks like|and before changes:|query plan examples:).*$', '', text)
    # Hapus link/url (http/https diikuti karakter non-spasi)
    text = re.sub(r'https?://\S+', ' ', text)
    

    # # Hapus baris upgrade dependency (misal: org.apache.ant:ant ....... 1.10.12 -> 1.10.13)
    # text = re.sub(r'^[\w\.\-]+:[\w\.\-]+[ .]*\d+(\.\d+)*\s*->\s*\d+(\.\d+)*$', ' ', text, flags=re.MULTILINE)
    # # Hapus baris yang hanya pattern "package ....... 1.2.3 -> 1.2.4"
    # text = re.sub(r'^\s*[\w\.\-]+[ .]*\d+(\.\d+)*\s*->\s*\d+(\.\d+)*$', ' ', text, flags=re.MULTILINE)


    # # Hapus baris log/tool output seperti [INFO] ...
    # text = re.sub(r'^\[info\].*$', ' ', text, flags=re.MULTILINE)


    # Hapus blok diff patch (mulai dari diff --git sampai sebelum narasi atau habis)
    # text = re.sub(r'diff --git[\s\S]*?(?=(?:\n[^\+\-\@\s])|$)', ' ', text)
    # # Hapus baris patch (---, +++, @@, -xxx, +xxx)
    # text = re.sub(r'(?m)^(---|\+\+\+|@@|\+|\-).*$\n?', ' ', text)

    # # Hapus blok log diapit {{ ... }}
    # text = re.sub(r'\{\{.*?\}\}', ' ', text, flags=re.DOTALL)



    # Hapus blok kode Java (dari import ... hingga penutup kode})
    # text = re.sub(r'(?s)import .+?^\}', ' ', text, flags=re.MULTILINE)
    # Hapus baris dengan indentasi kode (4+ spasi/tab)
    # text = re.sub(r'(?:\n|^)(    |\t).*(?:\n|$)', '\n', text)


    # Hapus blok kode Java (dari package ... hingga penutup kurung}) -- apakah perlu ini, atau ini membuat agresif?
    # text = re.sub(r'(?s)package [\s\S]+?^\}', '', text, flags=re.MULTILINE)
    # Hapus baris dengan indentasi kode (4 spasi/tab)
    # text = re.sub(r'(?:\n|^)(    |\t).*(?:\n|$)', '\n', text)

    # text = re.sub(r"drill physical[\s\S]*", '', text, flags=re.IGNORECASE)

    # # Hapus Semua Karakter Non-ASCII
    # text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Hapus baris kosong
    text = re.sub(r'\n+', ' ', text)
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text




# Terapkan pada field "Summary_Description"
df["Summary_Description"] = df["Summary_Description"].apply(clean_jira_text)



# === Filter teks terlalu pendek/panjang ===
df["text_len"] = df["Summary_Description"].apply(lambda x: len(x.split()))
df = df[(df["text_len"] > 2) & (df["text_len"] < 512)]
df = df.drop(columns=["text_len"])

# === Encode labels (label: ATD/Non-ATD → 1/0) ===
# label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
# df["label"] = df["label"].map(label_mapping)

# === Encode labels explicitly: 'ATD' → 1, 'Non-ATD' → 0 ===
# label_mapping = {'Non-ATD': 0, 'ATD': 1}
# df["label"] = df["label"].map(label_mapping)


# === Simpan hasil preprocessing (optional)
df.to_csv("/scratch/p311371/al-results/0-1-LATEST-ATD-DATASET-CLEANED-20M-WeaktoATD.csv", index=False)

# === Ambil teks dan label ===
texts = df["Summary_Description"].values
labels = df["label"].values

# === Stratified Split ===
train_text, test_text, train_idx, test_idx = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# === Parameter Active Learning ===
query_size = 100           # jumlah data awal
num_queries = 25           # iterasi active learning
query_strategy = RandomSampling()
transformer_model = "bert-base-uncased" #"bert-base-uncased"
filepath = "/scratch/p311371/al-results/"
num_classes = 2
batch_size = 32
num_epochs = 5

# === Jalankan Active Learning ===
labelled_df, results_df = fullpipeline(
    train_idx, train_text,
    test_idx, test_text,
    query_size, num_queries,
    query_strategy, transformer_model,
    filepath, num_classes,
    batch_size, num_epochs
)

# === Simpan hasil ===
results_df.to_csv(filepath + "0-1-YAL_active_learning_results_PROCESSED.csv", index=False)
print("\nActive Learning Process Completed.")
