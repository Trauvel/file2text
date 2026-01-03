from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import os

# Папка с файлами текста, расшифрованного из аудио
TEXT_DIR = 'text'

# Собираем все сегменты текста
all_segments = []
for filename in os.listdir(TEXT_DIR):
    if filename.endswith("_full.txt"):
        with open(os.path.join(TEXT_DIR, filename), "r", encoding="utf-8") as file:
            text = file.read()
            segments = text.split(". ")  # Разделение на сегменты (по предложениям)
            all_segments.extend(segments)

# Создаем модель BERTopic
topic_model = BERTopic(language="russian")

# Обучаем модель на сегментах текста
topics, probs = topic_model.fit_transform(all_segments)

# Получаем информацию о темах и их важность
topic_info = topic_model.get_topic_info()
print("Найденные темы:")
print(topic_info)

# Кластеризация сегментов по темам и создание логических блоков
topics_dict = {}
for idx, topic in enumerate(topics):
    if topic not in topics_dict:
        topics_dict[topic] = []
    topics_dict[topic].append(all_segments[idx])

# Сохраняем логические блоки в файлы
LOGICAL_BLOCKS_DIR = "logical_blocks"
os.makedirs(LOGICAL_BLOCKS_DIR, exist_ok=True)

for topic, segments in topics_dict.items():
    topic_name = f"Тема_{topic}" if topic != -1 else "Разные_темы"
    with open(os.path.join(LOGICAL_BLOCKS_DIR, f"{topic_name}.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(segments))

print(f"Логические блоки сохранены в папке {LOGICAL_BLOCKS_DIR}")