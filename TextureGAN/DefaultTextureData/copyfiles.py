import os
from PIL import Image

def rotate_and_save(src_path, dest_path, index):
    with Image.open(src_path) as img:
        r0=img
        r1=img.rotate(-90)  # -90 degrees for a clockwise rotation
        r2=img.rotate(-180)  # -180 degrees for a clockwise rotation
        r3=img.rotate(-270)  # -270 degrees for a clockwise rotation
        r0.save(f"{dest_path}/{index}.png")
        r1.save(f"{dest_path}/{index+1}.png")
        r2.save(f"{dest_path}/{index+2}.png")
        r3.save(f"{dest_path}/{index+3}.png")

def list_all_files(dir_path):
    file_paths = []

    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if ".DS_Store" not in filename:
                file_paths.append(os.path.join(dirpath, filename))

    return file_paths

full_path = os.path.abspath(__file__)
base_path = os.path.dirname(full_path)
pathto=f"{base_path}/data/sel"
save_path=f"{base_path}/used_data"
files=list_all_files(pathto)
print(files)
for index,file in enumerate(files):
    rotate_and_save(file,save_path,index*4)



