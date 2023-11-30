full_split_file = './splits/eigen_full/train_files.txt'

# 打开文件并读取所有行
with open(full_split_file, 'r') as file:
    lines = file.readlines()

# 创建一个新的空行列表
new_lines = []

# 遍历每一行
for line in lines:
    # 检查每一行的最后一个字符是否为 'l'
    if line.strip().split()[-1] == 'l':
        # 如果是，将该行添加到新的行列表中
        new_lines.append(line)

# 将新的行列表写回到文件中
with open(full_split_file, 'w') as file:
    file.writelines(new_lines)