import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Tên các cột dựa theo UCI Dataset
columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]

# Đọc dữ liệu từ file (cần đặt đúng tên file trong cùng thư mục)
df = pd.read_csv("mushroom/agaricus-lepiota.data", header=None)

# Xem 5 dòng đầu tiên
print("5 dòng đầu:")
print(df.head())

# Thông tin tổng quan
print("\nThông tin dataset:")
print(df.info())

# Thống kê nhanh về giá trị duy nhất trong mỗi cột
print("\nGiá trị duy nhất trong mỗi cột:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")