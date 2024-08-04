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
from concurrent.futures import ProcessPoolExecutor


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
    merged_topics = topics.copy()
    for i in range(len(topics)):
        if merged_topics[i] == "":
            continue
        for j in range(i + 1, len(topics)):
            if merged_topics[j] == "":
                continue
            if similarity_matrix[i, j] >= threshold:
                merged_topics[j] = merged_topics[i]

    return merged_topics


# Function to visualize the topic occurrences bar chart
def visualize_topic_occurrences(df):
    topic_counts = df["merged_label_names"].value_counts().head(10)

    plt.figure(figsize=(12, 8))
    topic_counts.sort_values(ascending=False).plot(kind="bar")
    plt.title("Occurrences of Topics")
    plt.xlabel("Merged Topic Names")
    plt.ylabel("Number of Occurrences")
    plt.xticks(rotation=75, ha="right", fontsize=10)  # Adjust font size and rotation
    plt.tight_layout()
    plt.savefig(
        "topic_occurrences.png", bbox_inches="tight"
    )  # Use bbox_inches to ensure labels fit in the image
    plt.show()
    return


# Function to visualize the topic occurrences pie chart
def visualize_topic_occurrences_pie(df):
    # Select top 10 topics by occurrence
    topic_counts = df["merged_label_names"].value_counts().head(10)

    # Create a pie chart
    plt.figure(figsize=(12, 8))
    plt.pie(
        topic_counts,
        labels=topic_counts.index,
        autopct="%1.1f%%",
        textprops={"fontsize": 10},
    )
    plt.title("Occurrences of Common Topics")
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig("topic_occurrences_pie.png", bbox_inches="tight")
    plt.show()
    return


# Parallel text processing
def process_text(row):
    combined_text = f"{row['invention_title']['text']} {row['abstract']['text']} {row['claims']['text']}"
    cleaned_text = clean_text(combined_text)
    return remove_stopwords(cleaned_text)


def main():
    if len(sys.argv) == 1:
        print("=== No data folder provided ===")
        exit(0)
    data_folder = sys.argv[1] + "/"

    input_file = data_folder + "case_data.txt"
    output_file = data_folder + "case_data.json"

    if not os.path.exists(output_file):
        print("📌 Converting into Json format..")
        format_and_save_json(input_file, output_file)

    patent_data = output_file

    print("=== Loading patent data ===")

    df = pd.read_json(patent_data)

    # print("🟢 Sample patent data:", df.head(1))

    print("=== Pre-processing patent data ===")

    with ProcessPoolExecutor() as executor:
        df["cleaned_text"] = list(
            executor.map(process_text, df.to_dict(orient="records"))
        )

    print("=== Loading BERTopic model ===")

    topic_model = BERTopic()
    df["topic"] = topic_model.fit_transform(df["cleaned_text"].tolist())[0]

    topic_labels = topic_model.generate_topic_labels()
    topic_labels = [label for label in topic_labels if "-1_" not in label]
    embeddings = topic_model.topic_embeddings_

    print("=== Finding Topic Similarities ===")

    similarity_matrix = cosine_similarity(embeddings)
    merged_topics = merge_similar_topics(topic_labels, similarity_matrix)

    # eleminate the same words in merged topics
    merged_topics = list(set(merged_topics))

    df_topics = pd.DataFrame(merged_topics)

    df_topics.to_csv(data_folder + "extracted_topics.csv", index=False)

    topic_labels_df = pd.DataFrame({"Label": merged_topics})

    new_columns = {}

    for index, row in topic_labels_df.iterrows():
        label = row["Label"]
        label_words = label.split("_")
        column_data = df["cleaned_text"].apply(
            lambda x: any(re.search(word, x, re.IGNORECASE) for word in label_words)
        )
        new_columns[f"topic_{index}"] = column_data

    new_columns_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_columns_df], axis=1)

    df["true_topic_nums"] = df.iloc[:, 15:].apply(
        lambda row: [i for i, val in enumerate(row) if val], axis=1
    )

    df["merged_label_names"] = df["true_topic_nums"].apply(
        lambda topic_nums: ", ".join(
            merged_topics[i] for i in topic_nums if i < len(merged_topics)
        )
    )

    print(
        "🟢 Mostly occured topics:",
        df["merged_label_names"].value_counts().head(1),
    )

    print("=== Visualizing topic occurances ===")

    # visualize_topic_occurrences(df)
    visualize_topic_occurrences_pie(df)

    print("=== Done ===")
    exit(0)


if __name__ == "__main__":
    main()
