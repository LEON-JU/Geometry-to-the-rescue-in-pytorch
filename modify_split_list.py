'''
这个脚本的作用是把full eigen list文件中我没有下载的部分给去除，使得训练时只使用我下载的那部分数据集
'''


# 读取完整的split list文件
full_split_file = './splits/eigen_full(backup)/val_files.txt'

# 已下载文件夹的列表
downloaded_folders = [
    '2011_09_26/2011_09_26_drive_0002_sync',
    '2011_09_26/2011_09_26_drive_0005_sync',
    '2011_09_26/2011_09_26_drive_0009_sync',
    '2011_09_26/2011_09_26_drive_0011_sync',
    '2011_09_26/2011_09_26_drive_0013_sync',
    '2011_09_26/2011_09_26_drive_0014_sync',
    '2011_09_26/2011_09_26_drive_0017_sync',
    '2011_09_26/2011_09_26_drive_0018_sync',
    '2011_09_26/2011_09_26_drive_0048_sync',
    '2011_09_26/2011_09_26_drive_0051_sync',
    '2011_09_26/2011_09_26_drive_0056_sync',
    '2011_09_26/2011_09_26_drive_0057_sync',
    '2011_09_26/2011_09_26_drive_0060_sync',
    '2011_09_26/2011_09_26_drive_0084_sync',
    '2011_09_26/2011_09_26_drive_0091_sync',
    '2011_09_26/2011_09_26_drive_0093_sync',
    '2011_09_26/2011_09_26_drive_0015_sync',
    '2011_09_26/2011_09_26_drive_0027_sync',
    '2011_09_26/2011_09_26_drive_0028_sync',
    '2011_09_26/2011_09_26_drive_0029_sync',
    '2011_09_26/2011_09_26_drive_0095_sync',
    '2011_09_26/2011_09_26_drive_0096_sync',
    '2011_09_26/2011_09_26_drive_0104_sync',
    '2011_09_26/2011_09_26_drive_0106_sync',
    '2011_09_26/2011_09_26_drive_0113_sync',
    '2011_09_26/2011_09_26_drive_0117_sync'
]

# 读取已下载的文件夹名字，用于后续比较
downloaded_folders_set = set(downloaded_folders)

# 读取full split list文件内容并进行处理
with open(full_split_file, 'r') as file:
    lines = file.readlines()

# 存储最终结果的列表
filtered_lines = []

# 遍历split list的每一行，检查是否在已下载文件夹列表中
for line in lines:
    folder_name = line.split()[0]  # 获取文件夹名字部分
    if folder_name in downloaded_folders_set:
        filtered_lines.append(line)  # 如果在已下载列表中，保留该行

# 将筛选后的结果写入新文件中
filtered_split_file = './splits/eigen_full/val_files.txt'  # 新文件路径
with open(filtered_split_file, 'w') as file:
    file.writelines(filtered_lines)
