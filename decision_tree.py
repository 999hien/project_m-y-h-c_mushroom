import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Đọc dữ liệu và xử lý
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]

df = pd.read_csv("mushroom/agaricus-lepiota.data", header=None, names=columns)

# Loại bỏ cột veil-type (chỉ có 1 giá trị)
df = df.drop(columns=['veil-type'])

# Thay thế giá trị thiếu "?" bằng NaN → loại bỏ
df = df.replace('?', pd.NA)
df = df.dropna()

# Mã hóa tất cả các biến phân loại thành số
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# 2. Tách dữ liệu
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Huấn luyện mô hình Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. Đánh giá mô hình
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))
