import streamlit as st
import pickle
import pandas as pd

# Tải mô hình cây quyết định
model_filename = './output/model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Tiêu đề giao diện
st.title("Dự đoán với Mô Hình Cây Quyết Định")

# Nhập dữ liệu đầu vào
st.sidebar.header("Thông tin đầu vào")
feature_1 = st.sidebar.number_input("Giá trị của đặc trưng 1:", min_value=0.0, value=0.0)
feature_2 = st.sidebar.number_input("Giá trị của đặc trưng 2:", min_value=0.0, value=0.0)
# Thêm nhiều đặc trưng nếu cần thiết

# Tạo DataFrame cho đầu vào
input_data = pd.DataFrame({
    'feature_1': [feature_1],
    'feature_2': [feature_2],
    # Thêm các đặc trưng khác vào đây
})

# Dự đoán
if st.sidebar.button("Dự đoán"):
    prediction = model.predict(input_data)
    st.write("Dự đoán của mô hình là:", prediction[0])

# Chạy ứng dụng
if __name__ == "__main__":
    st.write("Nhập thông tin để xem dự đoán của mô hình.")
