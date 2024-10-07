import pickle

# Đường dẫn tới file pickle
file_path = r'D:\ttnt\output\model.pkl'

# Mở file và đọc dữ liệu
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Hiển thị dữ liệu đã đọc
print(data)
