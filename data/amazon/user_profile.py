# %%
import os
import pickle
import numpy as np
import json
from tqdm import tqdm
import jsonlines
import copy
import matplotlib.pyplot as plt
import requests

# %%
dataset = "amazon"
inter_file = "cloth_sport"
all_file = "itm_emb_np"
domainA_file = "{}_A".format(all_file) # domain A file
domainB_file = "{}_B".format(all_file) # domain B file
N_cluster = 10  # the number of clusters

# %%
llm_emb_A = pickle.load(open(os.path.join("./handled/", 
                                          "{}.pkl".format(domainA_file)), "rb"))
llm_emb_B = pickle.load(open(os.path.join("./handled/", 
                                          "{}.pkl".format(domainB_file)), "rb"))
llm_emb = np.concatenate([llm_emb_A, llm_emb_B])

# %%
inter_seq, domain_seq = pickle.load(open('./handled/%s.pkl' % inter_file, 'rb'))
id_map = json.load(open("./handled/id_map.json"))
item_num_dict = id_map["item_dict"]["item_count"]

# %%
user_inter = {}

for user in tqdm(inter_seq.keys()):
    
    meta_seq = copy.deepcopy(inter_seq[user][:-1])
    meta_domain = domain_seq[user][:-1]

    meta_seq[meta_domain==1] = meta_seq[meta_domain==1] + item_num_dict["1"]

    user_inter[user] = meta_seq

# %% [markdown]
# ### Cluster

# %%
## 聚类
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

model = KMeans(n_clusters=N_cluster)
model.fit(llm_emb)

# %%
yhat = model.predict(llm_emb)
clusters = np.unique(yhat)
cluster_index = []
for cluster in clusters: # get the index of each cluster
    cluster_index.append(np.where(yhat == cluster)[0])

# %%
# 确保聚类的正确性
assert len(np.unique(np.concatenate(cluster_index))) == item_num_dict["0"]+item_num_dict["1"]

# %%
## Construct the map from item id to cluster id
cluster_map = {}
for cluster_id, index_list in enumerate(cluster_index):
    for item_id in index_list:
        cluster_map[item_id+1] = cluster_id

# %%
min(cluster_map.keys()), max(cluster_map.keys())

# %% [markdown]
# ### Partition the sequence

# %%
partitioned_user_inter = {}
for user, inter in user_inter.items():
    partition_inter = [[] for _ in range(N_cluster)]
    for item_id in inter:
        partition_inter[cluster_map[item_id]].append(item_id)
    partitioned_user_inter[user] = partition_inter

# %%
partition_len_list = []
for partition_inter in partitioned_user_inter.values():
    for inter_list in partition_inter:
        if len(inter_list) > 0:
            partition_len_list.append(len(inter_list))

# %%
np.mean(partition_len_list)

# %%
plt.hist(partition_len_list)

# %%
len_list = []
for inter_list in inter_seq.values():
    len_list.append(len(inter_list))

# %%
np.mean(len_list)

# %%
plt.hist(len_list)

# %% [markdown]
# ### Generate User Profile

# %%
url = ""

payload = json.dumps({
   "model": "gpt-3.5-turbo-ca",
   "messages": [
       {
           "role": "user",
           "content": "Assume you are a consumer who is shopping online. What's your interests? Please conclude it not beyond 100 words."
           },
   ],
   "temperature": 0.5,
   "max_tokens": 1024,
   "top_p": 1,
})
headers = {
   'Authorization': '',
   'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
   'Content-Type': 'application/json'
}

# response = requests.request("POST", url, headers=headers, data=payload)

# print(response.text)

# %%
response = requests.request("POST", url, headers=headers, data=payload)
re_json = json.loads(response.text)

# %%
re_json["choices"]

# %%
response = requests.request("POST", url, headers=headers, data=payload)

# %%
def get_response(prompt):
    url = ""

    payload = json.dumps({
    "model": "gpt-3.5-turbo-ca",
    "messages": [
       {
           "role": "user",
           "content": prompt
           },
    ],
    "temperature": 0.5,
    "max_tokens": 1024,
    "top_p": 1,
    })
    headers = {
    'Authorization': '',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    re_json = json.loads(response.text)

    return re_json["choices"][0]["message"]["content"], response

# %%
title_A = pickle.load(open("./handled/title_A.pkl", "rb"))
title_B = pickle.load(open("./handled/title_B.pkl", "rb"))
title_list = title_A + title_B

# %%
prompt_template = {
    "summarizer": "Assume you are an consumer and there are preference demonstrations from several aspects are as follows:\n {}. \n\n Please illustrate your preference with less than 100 words",
    "analyzer": "Assume you are a consumer who is shopping online. You have shown interests in following commdities:\n {}. \n\n The commodities are segmented by '\n'. \n\n Please conclude it not beyond 50 words. Do not only evaluate one specfic commodity but illustrate the interests overall."
}

# %%
test_user_inter = {}    # test code
for i in range(1, 2):
    test_user_inter[i] = partitioned_user_inter[i]

# %%
test_user_inter.keys()

# %%
if os.path.exists("./handled/user_profile.pkl"):
    user_profile = pickle.load(open("./handled/user_profile.pkl", "rb"))
else:
    user_profile = {}

while 1:
    try:
        for i, (user, partition_inter) in enumerate(tqdm(partitioned_user_inter.items())):
            
            if user in user_profile.keys():
                continue
            
            partition_pref = []
            for meta_inter in partition_inter:
                if len(meta_inter) > 0:
                    if len(meta_inter) > 15:    # avoid too long sequence
                        temp_meta_inter = meta_inter[-15:]
                    else:
                        temp_meta_inter = meta_inter
                    inter_str = ""
                    for item_id in temp_meta_inter:
                        inter_str = inter_str + title_list[item_id-1] + " \n"
                    pref_prompt = copy.deepcopy(prompt_template["analyzer"])
                    pref_prompt = pref_prompt.format(inter_str)
                    pref, response = get_response(pref_prompt)
                    partition_pref.append(pref)

            all_pref = ""
            for meta_pref in partition_pref:
                all_pref = all_pref + meta_pref + " \n"
            summary_prompt = copy.deepcopy(prompt_template["summarizer"])
            summary_prompt = summary_prompt.format(all_pref)
            summary, _ = get_response(summary_prompt)

            user_profile[user] = summary

            if i % 100 == 0:    # checkpoint
                with open("./handled/user_profile.pkl", "wb") as f:
                    pickle.dump(user_profile, f)
    except:
        with open("./handled/user_profile.pkl", "wb") as f:
            pickle.dump(user_profile, f)

    if len(user_profile) == len(partitioned_user_inter):
        break
    

def get_embedding(prompt):
    url = ""

    payload = json.dumps({
    "model": "text-embedding-ada-002",
    "input": prompt
    })
    headers = {
    'Authorization': '',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    re_json = json.loads(response.text)

    return re_json["data"][0]["embedding"]


user_emb = {}

# %%
while 1:    # avoid broken due to internet connection
    try:
        for key, value in tqdm(user_profile.items()):
            if key not in user_emb.keys():
                user_emb[key] = get_embedding(value)
    except:
        continue
    if len(user_emb) == len(user_profile):
        break

# %%
len(user_emb)

# %%
emb_list = []
for key, value in tqdm(user_emb.items()):
    emb_list.append(value)

emb_list = np.array(emb_list)
pickle.dump(emb_list, open("./handled/usr_profile_emb.pkl", "wb"))

