#!/usr/bin/env python3
import numpy as np

attn_score = np.load("/home/kwu/HET/hrt/attn_score_dump.npy")
with open("attn_score_dump.out", "w") as fd:
    for i in range(len(attn_score)):
        fd.write(str(float(attn_score[i])) + "\n")
