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

# Nhập các đặc trưng từ giao diện người dùng
age = st.sidebar.number_input("Tuổi:", min_value=18, max_value=100, value=34)
no_of_days_subscribed = st.sidebar.number_input("Số ngày đã đăng ký:", min_value=1, max_value=1000, value=107)
weekly_mins_watched = st.sidebar.number_input("Số phút xem hàng tuần:", min_value=0.0, value=201.0)
minimum_daily_mins = st.sidebar.number_input("Số phút xem ít nhất mỗi ngày:", min_value=0.0, value=7.3)
maximum_daily_mins = st.sidebar.number_input("Số phút xem nhiều nhất mỗi ngày:", min_value=0.0, value=22.78)
weekly_max_night_mins = st.sidebar.number_input("Số phút xem tối đa ban đêm hàng tuần:", min_value=0.0, value=79.0)
videos_watched = st.sidebar.number_input("Số lượng video đã xem:", min_value=0, max_value=100, value=3)
maximum_days_inactive = st.sidebar.number_input("Số ngày không hoạt động tối đa:", min_value=0, max_value=30, value=2)
customer_support_calls = st.sidebar.number_input("Số lần gọi hỗ trợ khách hàng:", min_value=0, max_value=20, value=3)

# Tạo DataFrame cho đầu vào
input_data = pd.DataFrame({
    'age': [age],
    'customer_support_calls': [customer_support_calls],
    'maximum_daily_mins': [maximum_daily_mins],
    'maximum_days_inactive': [maximum_days_inactive],
    'minimum_daily_mins': [minimum_daily_mins],
    'no_of_days_subscribed': [no_of_days_subscribed],
    'videos_watched': [videos_watched],
    'weekly_max_night_mins': [weekly_max_night_mins],
    'weekly_mins_watched': [weekly_mins_watched]
})

# Dự đoán
if st.sidebar.button("Dự đoán"):
    prediction = model.predict(input_data)
    result = "Khách hàng sẽ rời bỏ" if prediction[0] == 1 else "Khách hàng sẽ không rời bỏ"
    
    # Hiển thị kết quả với chữ lớn hơn
    st.markdown(f"<h2 style='text-align: center; color: red;'>{result}</h2>", unsafe_allow_html=True)


# Chạy ứng dụng
if __name__ == "__main__":
    st.write("Nhập thông tin để xem dự đoán của mô hình.")
