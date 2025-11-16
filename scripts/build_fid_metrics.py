import os, shutil
from torchvision.datasets import CIFAR10
from cleanfid import fid

def build_real_stats_per_class(cache_root, mode="clean"):
    os.makedirs(cache_root, exist_ok=True)
    ds = CIFAR10(root=os.path.join(cache_root, "data"), train=True, download=True)
    class_dirs = []
    # unpack dataset
    for c in range(10):
        d = os.path.join(cache_root, f"real_c{c}")
        class_dirs.append(d)
        if not os.path.exists(d) or len(os.listdir(d)) < 5000: 
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

    need_write = any(len(os.listdir(d)) == 0 for d in class_dirs)
    if need_write:
        counts = [0]*10
        for img, y in ds:
            cdir = class_dirs[y]
            save_path = os.path.join(cdir, f"{counts[y]:05d}.png")
            img.save(save_path)
            counts[y] += 1

    custom_names = {}
    for c in range(10):
        custom_name = f"cifar10_train_clean_class{c}"
        if not fid.test_stats_exists(custom_name, mode):
            fid.make_custom_stats(custom_name, class_dirs[c], mode=mode)
        custom_names[c] = custom_name

    return custom_names  

if __name__ == "__main__":
    build_real_stats_per_class("cache", mode="clean")
    