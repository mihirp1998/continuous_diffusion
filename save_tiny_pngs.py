from RAE.src.utils import basic_utils
import os
import ipdb
st = ipdb.set_trace
dataset_val = basic_utils.TextDataset(root="/home/mprabhud/datasets/")
for i in range(len(dataset_val)):
    image = dataset_val[i]
    os.makedirs(f"/home/mprabhud/datasets/tiny_rawdata", exist_ok=True)
    # st()
    image[0].save(f"/home/mprabhud/datasets/tiny_rawdata/{i:05d}.png")
    open(f"/home/mprabhud/datasets/tiny_rawdata/{i:05d}.txt", "w").write(image[1])
    if i > 10:
        break