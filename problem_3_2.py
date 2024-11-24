#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:42:30 2024

@author: tusker
"""

import numpy as np
import pandas as pd
from pybloom_live import BloomFilter
#%%
dataset=pd.read_csv("dataset_1M.csv", sep=";")
print("Data shape: " , dataset.shape )
items = dataset.iloc[:, 0] 

counts = dataset.iloc[:, 0].value_counts()
count_four_or_more = (counts >= 4).sum()
print(f"True count of items appearing four or more times: {count_four_or_more}")
#%%

n = len( dataset ) 
p = 0.01 

b = -n * np.log( p ) / ( np.log( 2 ) ** 2 )  
k = np.ceil( ( b / n ) * np.log( 2 ) )  # from lecture slide

#bloom_filter = BloomFilter( capacity = n, error_rate = p )
bloom_once = BloomFilter( capacity = n, error_rate = p )
bloom_twice = BloomFilter( capacity = n, error_rate = p )
bloom_thrice = BloomFilter( capacity = n, error_rate = p )

""" for item in dataset.iloc[ :, 0 ]:
    bloom_filter.add( item ) """
count_four_or_more = 0

for item in items:
    if item in bloom_thrice:
        count_four_or_more += 1
    elif item in bloom_twice:
        bloom_thrice.add(item)
    elif item in bloom_once:
        bloom_twice.add(item)
    else:
        bloom_once.add(item)

#%%
# print("Known item in Bloom Filter:", 82058 in bloom_filter)  
# print("Unknown item in Bloom Filter:", 9999999 in bloom_filter)

print(f"Number of items appearing four or more times: {count_four_or_more}")