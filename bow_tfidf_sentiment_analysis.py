import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# 📁 Output folder
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# 📝 Corpus
corpus = [
    "Raj eats burgers before cricket."

"Pizza is tasty after tennis."

"They play football on weekends."

"Burgers and fries are her favorite."

"He enjoys tennis and pizza."
]

# ✅ Bag-of-Words
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(corpus).toarray()
bow_vocab = bow_vectorizer.vocabulary_
print("\nVocabulary:\n", dict(sorted(bow_vocab.items(), key=lambda x: x[1])))
print("\nBoW Matrix:\n", bow_matrix)

# ✅ TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus).toarray()
tfidf_vocab = tfidf_vectorizer.vocabulary_
print("\nTF-IDF Matrix:\n", np.round(tfidf_matrix, 8))

# ✅ WordCloud for BoW with sentence number title
for i, sentence in enumerate(corpus):
    word_freq = dict(zip(bow_vectorizer.get_feature_names_out(), bow_matrix[i]))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

    plt.figure(figsize=(6, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"BoW WordCloud - Sentence {i+1}")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/bow_wordcloud_s{i+1}.png")
    plt.close()

# ✅ WordCloud for TF-IDF with sentence number title
for i, sentence in enumerate(corpus):
    word_freq = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_matrix[i]))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

    plt.figure(figsize=(6, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"TF-IDF WordCloud - Sentence {i+1}")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/tfidf_wordcloud_s{i+1}.png")
    plt.close()

# ✅ Cosine Similarity: BoW
bow_similarity = cosine_similarity(bow_matrix)
plt.figure(figsize=(6, 5))
sns.heatmap(bow_similarity, annot=True, cmap="Blues", fmt=".2f")
plt.title("Cosine Similarity - BoW")
plt.savefig(f"{output_folder}/bow_similarity.png")
plt.close()

# ✅ Cosine Similarity: TF-IDF
tfidf_similarity = cosine_similarity(tfidf_matrix)
plt.figure(figsize=(6, 5))
sns.heatmap(tfidf_similarity, annot=True, cmap="Greens", fmt=".2f")
plt.title("Cosine Similarity - TF-IDF")
plt.savefig(f"{output_folder}/tfidf_similarity.png")
plt.close()

# ✅ Final prints
print("\nCosine Similarity (BoW):\n", np.round(bow_similarity, 2))
print("\nCosine Similarity (TF-IDF):\n", np.round(tfidf_similarity, 2))
print("\n✅ All word clouds and similarity matrices saved in 'output/' folder.")