from pathlib import Path
from glob import glob
import pandas as pd

folder_loc = "outputs/2024-02-21/11-23-39"

topic_files = glob(folder_loc + "/topic_*.txt")
parent_file_by_topic = []
for topic_file in topic_files:
    with open(topic_file, "r") as file:
        lines = file.readlines()
        topic_group = []
        for line in lines:
            file_path = Path(line.strip())
            parent_name = file_path.parent.name
            topic_group.append(parent_name)
        parent_file_by_topic.append(topic_group)

parent_count_by_topic = {}
for i, topic in enumerate(parent_file_by_topic):
    value_counts = pd.Series(topic).value_counts()
    for parent, count in value_counts.items():
        if parent not in parent_count_by_topic:
            parent_count_by_topic[parent] = {}
        parent_count_by_topic[parent][i] = count

total_counts = 0
correct_counts = 0
for key, values in parent_count_by_topic.items():
    total_counts += sum(values.values())
    correct_counts += max(values.values())
accuracy = correct_counts * 100 / total_counts
print(f"Accuracy: {accuracy:.2f}%")
a = 1
