import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu (bạn cần thay đúng tên file nếu khác)
column_names = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]

df = pd.read_csv("mushroom/agaricus-lepiota.data", header=None, names=column_names)
df['class'] = df['class'].map({'e': 'edible', 'p': 'poisonous'})

# Các thuộc tính muốn vẽ
selected_features = [
    'odor',
    'bruises',
    'gill-size',
    'gill-color',
    'cap-shape',
    'spore-print-color',
    'ring-type',
    'stalk-root'
]

# Vẽ và lưu từng biểu đồ
for feature in selected_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=feature, hue='class')
    plt.title(f'{feature} vs class')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=30)
    
    # Lưu ảnh
    filename = f'{feature}_vs_class.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Đóng figure sau khi lưu để không bị chồng lệnh
