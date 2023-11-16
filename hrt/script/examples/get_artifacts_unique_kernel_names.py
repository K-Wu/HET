#!/usr/bin/env python3
import csv
import os

filenames = ["hgt_acm_dgl_bw.csv", "hgt_acm_dgl_fw.csv", "hgt_acm_dgl_opt.csv"]
filenames = list(map(lambda x: os.path.join("artifacts", x), filenames))

if __name__ == "__main__":
    for filename in filenames:
        print(filename)
        unique_kernel_names = set()
        for row_idx, row in enumerate(csv.reader(open(filename))):
            if row_idx == 0:
                continue  # skip header
            unique_kernel_names.add(row[0])
        print(len(unique_kernel_names))
        print("\n".join(unique_kernel_names))
        print("------------------")
