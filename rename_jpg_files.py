import os

# 遍历文件夹中的所有图片文件
image_folder = './imgs/jidu/Sunny/'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# 生成新的文件名并重命名图片文件
new_image_files = []
for i, image_file in enumerate(image_files):
    new_image_file = f"{str(i).zfill(5)}.png"
    new_image_path = os.path.join(image_folder, new_image_file)
    old_image_path = os.path.join(image_folder, image_file)
    os.rename(old_image_path, new_image_path)
    new_image_files.append(new_image_file)

# 将原始文件名和新文件名写入txt文件
txt_file_path = os.path.join(image_folder, 'file_mapping.txt')
with open(txt_file_path, 'w') as f:
    for i, image_file in enumerate(image_files):
        f.write(f"{image_file} -> {new_image_files[i]}\n")