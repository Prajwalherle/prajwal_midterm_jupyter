#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import os


# In[3]:


print(os.getcwd())


# In[4]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from itertools import combinations
import time

# Define the path to the CSV file (updated path)
file_path = r"D:\NJIT\2nd sem NJIT_Spring\Data mining\pp958_Midterm_Jupyter\prajwal_mid\grocery_transactions.csv"

# Function to read integrated transactions
def read_integrated_transactions(file):
    with open(file, 'r') as f:
        content = f.read().splitlines()

    unique_items = []
    datasets = {}
    current_store = None

    for line in content:
        line = line.strip()

        if line == "Unique Items":
            continue  # Skip the header for unique items
        elif "Transactions" in line:
            if current_store is not None:
                current_store = line.replace(" Transactions", "").strip()
                datasets[current_store] = []
            else:
                current_store = line.replace(" Transactions", "").strip()
                datasets[current_store] = []
        elif current_store and line:
            cleaned_line = line.replace('"', '').strip()
            datasets[current_store].append(cleaned_line.split(','))

    return unique_items, datasets

# Brute force frequent itemsets
def brute_force_frequent_itemsets(data, min_support):
    itemsets = []
    total_transactions = len(data)
    for i in range(1, len(data.columns) + 1):
        for combo in combinations(data.columns, i):
            support_count = data[list(combo)].all(axis=1).sum()
            support = support_count / total_transactions
            if support >= min_support:
                itemsets.append((list(combo), support))
    return pd.DataFrame(itemsets, columns=['itemsets', 'support'])

# Process function
def process_store(transactions, min_support, min_confidence):
    encoder = TransactionEncoder()
    onehot = encoder.fit(transactions).transform(transactions)
    df = pd.DataFrame(onehot, columns=encoder.columns_)

    min_support_fraction = min_support / 100
    min_confidence_fraction = min_confidence / 100

    start_time = time.time()
    brute_force_freq_itemsets = brute_force_frequent_itemsets(df, min_support_fraction)
    print("\nBrute Force Frequent Itemsets:")
    display(brute_force_freq_itemsets)

    if not brute_force_freq_itemsets.empty:
        brute_force_rules = association_rules(brute_force_freq_itemsets, metric="confidence", min_threshold=min_confidence_fraction)
        print("\nBrute Force Association Rules:")
        display(brute_force_rules)

    start_time_apriori = time.time()
    frequent_itemsets_apriori = apriori(df, min_support=min_support_fraction, use_colnames=True)
    print("\nApriori Frequent Itemsets:")
    display(frequent_itemsets_apriori)

    if not frequent_itemsets_apriori.empty:
        rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence_fraction)
        print("\nApriori Association Rules:")
        display(rules_apriori)

    print("\nExecution Time:")
    print("Brute Force: {:.4f} seconds".format(time.time() - start_time))
    print("Apriori: {:.4f} seconds".format(time.time() - start_time_apriori))

# Load datasets
unique_items, datasets = read_integrated_transactions(file_path)

# List available stores
print("Available Stores:")
for i, name in enumerate(datasets.keys()):
    print(f"{i + 1}: {name}")

# Example values:
store_index = 0  # Change to 1, 2, etc. based on the printed list above
min_support = 20  # As a percentage
min_confidence = 50  # As a percentage

store_name = list(datasets.keys())[store_index]
print(f"\nSelected store: {store_name}")
print(f"\nTransactions for selected store:")
display(pd.DataFrame(datasets[store_name]))

# Process the store
process_store(datasets[store_name], min_support, min_confidence)


# In[ ]:




