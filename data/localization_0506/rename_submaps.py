import os
import glob

def rename_submaps():
    """
    将所有submap_x_updated.bin文件重命名为submap_x.bin
    """
    # 获取当前目录下所有的submap_x_updated.bin文件
    pattern = "submap_*_updated.bin"
    files = glob.glob(pattern)
    
    if not files:
        print("没有找到任何submap_x_updated.bin文件")
        return
    
    print(f"找到 {len(files)} 个文件需要重命名:")
    
    # 对文件进行排序，确保按数字顺序处理
    files.sort(key=lambda x: int(x.split('_')[1]))
    
    renamed_count = 0
    for old_name in files:
        # 提取数字部分
        parts = old_name.split('_')
        if len(parts) >= 3:
            number = parts[1]
            new_name = f"submap_{number}.bin"
            
            try:
                # 检查新文件名是否已存在
                if os.path.exists(new_name):
                    print(f"警告: {new_name} 已存在，跳过重命名 {old_name}")
                    continue
                
                # 重命名文件
                os.rename(old_name, new_name)
                print(f"重命名: {old_name} -> {new_name}")
                renamed_count += 1
                
            except Exception as e:
                print(f"重命名 {old_name} 时出错: {e}")
    
    print(f"\n完成! 成功重命名了 {renamed_count} 个文件")

if __name__ == "__main__":
    rename_submaps() 