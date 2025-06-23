import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# 🔧 Step 0: Create output folder
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📥 Step 1: Load dataset
df = pd.read_csv("IMDB 123.csv", encoding='utf-8-sig')
df.columns = df.columns.str.strip()
print("📋 Available Columns:", df.columns.tolist())

# ✅ Step 2: Set column names from your dataset
TEXT_COL = 'review'
LABEL_COL = 'sentiment'

# 🧹 Step 3: Clean data
df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].str.strip()
df[LABEL_COL] = df[LABEL_COL].str.strip()

# 🗃 Step 4: Filter rare labels
label_counts = df[LABEL_COL].value_counts()
valid_classes = label_counts[label_counts >= 2].index.tolist()
df = df[df[LABEL_COL].isin(valid_classes)]

# 🔢 Step 5: Encode class labels
label_encoder = LabelEncoder()
df['LabelEncoded'] = label_encoder.fit_transform(df[LABEL_COL])

# ✂ Step 6: Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL], df['LabelEncoded'], test_size=0.2, random_state=42, stratify=df['LabelEncoded']
)

# 🔁 Step 7: Pipelines for BoW and TF-IDF
cbow_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

tfidf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 🎯 Step 8: Train models
cbow_pipeline.fit(X_train, y_train)
tfidf_pipeline.fit(X_train, y_train)

# 🔮 Step 9: Make predictions
cbow_preds = cbow_pipeline.predict(X_test)
tfidf_preds = tfidf_pipeline.predict(X_test)

# ✅ Step 10: Metrics
used_labels = sorted(list(set(y_test)))
used_class_names = label_encoder.inverse_transform(used_labels)

cbow_accuracy = accuracy_score(y_test, cbow_preds) * 100
tfidf_accuracy = accuracy_score(y_test, tfidf_preds) * 100

cbow_precision = precision_score(y_test, cbow_preds, average='weighted') * 100
tfidf_precision = precision_score(y_test, tfidf_preds, average='weighted') * 100

cbow_recall = recall_score(y_test, cbow_preds, average='weighted') * 100
tfidf_recall = recall_score(y_test, tfidf_preds, average='weighted') * 100

cbow_f1 = f1_score(y_test, cbow_preds, average='weighted') * 100
tfidf_f1 = f1_score(y_test, tfidf_preds, average='weighted') * 100

# 🎉 Accuracy Print Summary
print(f"\nBag-of-Words Accuracy: {cbow_accuracy:.2f}%")
print(f"TF-IDF Accuracy: {tfidf_accuracy:.2f}%")

# 📊 Metric Table
print("\nPerformance Comparison Table:")
print(f"{'Metric':<12} {'Bag-of-Words':<15} {'TF-IDF'}")
print(f"{'Accuracy':<12} {cbow_accuracy:.2f}%{'':<10} {tfidf_accuracy:.2f}%")
print(f"{'Precision':<12} {cbow_precision:.2f}%{'':<10} {tfidf_precision:.2f}%")
print(f"{'Recall':<12} {cbow_recall:.2f}%{'':<10} {tfidf_recall:.2f}%")
print(f"{'F1 Score':<12} {cbow_f1:.2f}%{'':<10} {tfidf_f1:.2f}%")

# 🌥 Step 11: Save WordCloud
text_data = " ".join(df[TEXT_COL])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of All Text Data", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"))
plt.close()

# 🧮 Step 12: Confusion Matrix Images
def plot_confusion(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred, labels=used_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=used_class_names,
                yticklabels=used_class_names, cmap="YlGnBu")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

plot_confusion(y_test, cbow_preds, "Bag-of-Words", "confusion_matrix_bow.png")
plot_confusion(y_test, tfidf_preds, "TF-IDF", "confusion_matrix_tfidf.png")

# 📈 Step 13: Metric Comparison Graph
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
cbow_scores = [cbow_accuracy, cbow_precision, cbow_recall, cbow_f1]
tfidf_scores = [tfidf_accuracy, tfidf_precision, tfidf_recall, tfidf_f1]

x = range(len(metrics))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x, cbow_scores, width=bar_width, label='Bag-of-Words', color='skyblue')
plt.bar([i + bar_width for i in x], tfidf_scores, width=bar_width, label='TF-IDF', color='lightgreen')

plt.xlabel("Evaluation Metrics")
plt.ylabel("Score (%)")
plt.title("Bag-of-Words vs TF-IDF - Performance Comparison")
plt.xticks([i + bar_width / 2 for i in x], metrics)
plt.ylim(0, 110)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "performance_comparison_graph.png"))
plt.close()

print("\n✅ All visual outputs saved in the 'output/' folder.")