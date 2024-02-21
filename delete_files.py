import os

folder_path = './imgs/jidu/evening_results'  # 替换为你的文件夹路径

for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        file_number = int(filename.split('_')[0])
        if file_number >= 200:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"已删除文件: {filename}")