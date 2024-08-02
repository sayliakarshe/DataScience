# bert_topic_extraction.py

import json
import sys
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


# Function to format and save JSON
def format_and_save_json(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        data = infile.read()

    objects = [obj for obj in data.split("\n") if obj.strip()]
    formatted_data = []

    for obj in objects:
        try:
            json_object = json.loads(obj)
            formatted_data.append(json_object)
        except json.JSONDecodeError as e:
            print(f"Error parsing object: {e}")
            continue

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(formatted_data, outfile, indent=4, ensure_ascii=False)


# Function to clean text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])


# Function to merge similar topics
def merge_similar_topics(topics, similarity_matrix, threshold=0.9):
    merged_topics = []
    seen = set()

    for i, topic in enumerate(topics):
        if i in seen:
            continue

        similar_topics = [i]
        for j in range(i + 1, len(topics)):
            if similarity_matrix[i][j] > threshold:
                similar_topics.append(j)
                seen.add(j)

        merged_topic = topics[i]
        merged_topics.append(merged_topic)
        seen.add(i)

    return merged_topics


# Function to visualize the topic occurrences
def visualize_topic_occurrences(df):
    topic_counts = df["merged_label_names"].value_counts()

    plt.figure(figsize=(12, 8))
    topic_counts.sort_values(ascending=False).plot(kind='bar')
    plt.title("Occurrences of Merged Topics")
    plt.xlabel("Merged Topic Names")
    plt.ylabel("Number of Occurrences")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("topic_occurrences.png")
    plt.show()


# Main function to run topic extraction and visualization
def main():
    if len(sys.argv) == 1:
        print("=== No data folder provided ===")
        exit(0)
    data_folder = sys.argv[1] + "/"

    input_file = data_folder + "case_data.txt"
    output_file = data_folder + "case_data.json"

    # if case_data.json is not present in output file the apply format__and_save_json
    if not os.path.exists(output_file):
        format_and_save_json(input_file, output_file)

    # format_and_save_json(input_file, output_file)



    patent_data = output_file
    df = pd.read_json(patent_data)
    df["text"] = (
        df["invention_title"].apply(lambda x: x["text"])
        + " "
        + df["abstract"].apply(lambda x: x["text"])
        + " "
        + df["claims"].apply(lambda x: x["text"])
    )
    df["cleaned_text"] = df["text"].apply(clean_text).apply(remove_stopwords)

    topic_model = BERTopic()
    df["topic"] = topic_model.fit_transform(df["cleaned_text"].tolist())[0]

    topic_labels = topic_model.generate_topic_labels()
    topic_labels = [label for label in topic_labels if "-1_" not in label]
    embeddings = topic_model.topic_embeddings_
    similarity_matrix = cosine_similarity(embeddings)
    merged_topics = merge_similar_topics(topic_labels, similarity_matrix)

    topic_labels_df = pd.DataFrame({"Label": merged_topics})
    new_columns = {}

    for index, row in topic_labels_df.iterrows():
        label = row["Label"]
        label_words = label.split("_")
        column_data = [False] * len(df)

        for word in label_words:
            column_data = [
                col or bool(re.search(word, text, re.IGNORECASE))
                for col, text in zip(column_data, df["cleaned_text"])
            ]

        new_columns[f"topic_{index}"] = column_data

    new_columns_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_columns_df], axis=1)

    true_topics = []
    for _, row in df.iterrows():
        true_topic_nums = [
            i for i, val in enumerate(row[df.columns[15:]]) if val == True
        ]
        true_topics.append(true_topic_nums)

    df["true_topic_nums"] = true_topics

    merged_label_names = []
    for _, row in df.iterrows():
        topic_nums = row["true_topic_nums"]
        label_names = [merged_topics[i] for i in topic_nums if i < len(merged_topics)]
        merged_label_names.append(", ".join(label_names))

    df["merged_label_names"] = merged_label_names

    visualize_topic_occurrences(df)

    return df


if __name__ == "__main__":
    main()
