import os
import random
from pathlib import Path


data_dir = Path("/media/mdt/PNY/zalo/speech/data/Train-Test-Data/feature/test")
ids = data_dir.glob("*")
ids = [i for i in ids if i.is_dir()]
samples = 50000
output_txt = "/media/mdt/PNY/zalo/speech/data/Train-Test-Data/veri_test.txt"
with open(output_txt, "w") as f:
    for _ in range(samples):
        # same_id
        if random.random() > 0.5:
            id = random.choice(ids)
            fns = list(id.glob("*.npy"))
            fn1, fn2 = random.sample(fns, 2)
            text = f"1 {fn1.parts[-2]}/{fn1.parts[-1][:-4]}.wav {fn2.parts[-2]}/{fn2.parts[-1][:-4]}.wav\n"
            f.write(text)
        
        else:
            id1, id2 = random.sample(ids, 2)
            fns1 = list(id1.glob("*.npy"))
            fn1 = random.choice(fns1)
            fns2 = list(id2.glob("*.npy"))
            fn2 = random.choice(fns2)
            text = f"0 {fn1.parts[-2]}/{fn1.parts[-1][:-4]}.wav {fn2.parts[-2]}/{fn2.parts[-1][:-4]}.wav\n"
            f.write(text)