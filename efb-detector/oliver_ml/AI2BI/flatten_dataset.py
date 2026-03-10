import os
import shutil

#shutil.rmtree("datasets/flattened_frames")
os.makedirs("datasets/flattened_frames")

for base, _, file_names in os.walk("datasets/unflattened_frames"):
    for name in file_names:
        if not name.endswith(".JPG"):
            continue
        path = os.path.join(base, name)
        dst_path = os.path.join("datasets/flattened_frames", name)
        shutil.copyfile(path, dst_path)