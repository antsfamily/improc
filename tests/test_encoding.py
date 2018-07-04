#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-06 21:39:43
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import improc as imp


path = "../data/demo.txt"

h = imp.HuffmanCoding(path)

output_path = h.compress()
print("Compressed file path: " + output_path)

decom_path = h.decompress(output_path)
print("Decompressed file path: " + decom_path)


print("========================")

for k in h.codes:
    print(k, h.codes[k])
print("------------------------")

for k in h.reverse_mapping:
    print(h.reverse_mapping[k], k)
print("========================")
