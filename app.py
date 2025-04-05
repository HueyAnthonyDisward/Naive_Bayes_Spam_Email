import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV hoặc nguồn dữ liệu có sẵn
df = pd.read_csv("cleaned_data.csv")  # Thay "data.csv" bằng đường dẫn file thực tế

# Loại bỏ các dòng có giá trị NaN trong cột 'text'
df = df.dropna(subset=['text'])

X = df['text']
y = df['label']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Áp dụng TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Huấn luyện mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Dự đoán
y_pred = model.predict(X_test_tfidf)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit UI
st.title("Spam Email Detection")
st.write(f'**Accuracy:** {accuracy:.4f}')
st.text("Nhập nội dung email để kiểm tra:")

user_input = st.text_area("Email nội dung:", "")
if st.button("Dự đoán"):
    if user_input:
        user_input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(user_input_tfidf)[0]
        result = "Spam" if prediction == 1 else "Không phải spam"
        st.write(f'**Dự đoán:** {result}')

# Hiển thị ma trận nhầm lẫn
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)
