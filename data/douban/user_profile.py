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
from zhipuai import ZhipuAI

# %%
dataset = "douban"
inter_file = "book_movie"
all_file = "item_emb"
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
profile_emb = []

for user in range(1, len(inter_seq.keys())+1):
    profile_emb.append(np.zeros(4096))
profile_emb = np.array(profile_emb)

pickle.dump(profile_emb, open("./handled/user_emb.pkl", "wb"))

# %%
user_inter = {}

for user in tqdm(inter_seq.keys()):
    
    meta_seq = copy.deepcopy(inter_seq[user][:-1])
    meta_domain = domain_seq[user][:-1]

    meta_seq[meta_domain==1] = meta_seq[meta_domain==1] + item_num_dict["0"]

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
for user, inter in tqdm(user_inter.items()):
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
client = ZhipuAI(api_key="") 

def get_response(prompt, role_prompt):
    
    response = client.chat.completions.create(
    model="glm-4-flash",  # 填写需要调用的模型编码
    messages=[
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": prompt}
    ],
)

    return response.choices[0].message.content

# %%
role_template = {
    "summarizer": "你是一个电影和书籍的爱好者。",
    "analyzer": "你是一个电影和书籍的爱好者。",
}
prompt_template = {
    "summarizer": "你有如下几方面的爱好: \n {}. \n\n 请使用不超过100个字描述你的爱好",
    "analyzer": "你对以下的电影或者书籍很感兴趣: \n {}. \n\n 请用不超过50个字总结你的兴趣。",
}

# %%
title_A = pickle.load(open("./handled/title_A.pkl", "rb"))
title_B = pickle.load(open("./handled/title_B.pkl", "rb"))
title_list = title_A + title_B

# %%
test_user_inter = {}    # test code
for i in range(1, 4):
    test_user_inter[i] = partitioned_user_inter[i]

# %%
if os.path.exists("./handled/user_profile.pkl"):
    user_profile = pickle.load(open("./handled/user_profile.pkl", "rb"))
else:
    user_profile = {}

# %%
# user_profile = {}

# %%
while 1:
    # try:
    for i, (user, partition_inter) in enumerate(tqdm(partitioned_user_inter.items())):
        
        if user in user_profile.keys():
            continue
        
        partition_pref = []
        for meta_inter in partition_inter:
            if len(meta_inter) > 0:
                if len(meta_inter) > 10:    # avoid too long sequence
                    temp_meta_inter = meta_inter[-10:]
                else:
                    temp_meta_inter = meta_inter
                inter_str = ""
                for item_id in temp_meta_inter:
                    inter_str = inter_str + title_list[item_id-1][0] + " \n"
                #     break
                # while 1:    # avoid 敏感内容
                #     count = 0
                #     inter_str = title_list[temp_meta_inter[count]-1]
                #     try:
                #         pref_prompt = copy.deepcopy(prompt_template["analyzer"])
                #         pref_prompt = pref_prompt.format(inter_str)
                #         pref  = get_response(pref_prompt, role_template["analyzer"])
                #     except:
                #         count += 1
                #         continue
                #     if pref_prompt:
                #         break
                pref_prompt = copy.deepcopy(prompt_template["analyzer"])
                pref_prompt = pref_prompt.format(inter_str)
                try:
                    pref  = get_response(pref_prompt, role_template["analyzer"])
                except:
                    continue
                partition_pref.append(pref)

        all_pref = ""
        for meta_pref in partition_pref:
            all_pref = all_pref + meta_pref + " \n"
        summary_prompt = copy.deepcopy(prompt_template["summarizer"])
        summary_prompt = summary_prompt.format(all_pref)
        try:
            summary = get_response(summary_prompt, role_template["summarizer"])
        except:
            summary = "用户的爱好未知。"

        user_profile[user] = summary

        if i % 100 == 0:    # checkpoint
            with open("./handled/user_profile.pkl", "wb") as f:
                pickle.dump(user_profile, f)
    
    # except:
    #     with open("./handled/user_profile.pkl", "wb") as f:
    #         pickle.dump(user_profile, f)
    #     continue

    if len(user_profile) == len(partitioned_user_inter):
        break
    

# %%
pref_prompt

# %%
def get_embedding(prompt):
    
    response = client.embeddings.create(
        model="embedding-3", #填写需要调用的模型编码
        input=prompt
    )
    return response.data[0].embedding

# %%
if os.path.exists("./handled/usr_profile_emb.pkl"):
    user_emb = pickle.load(open("./handled/usr_profile_emb.pkl", "rb"))
else:
    user_emb = {}

# %%
i=0
while 1:    # avoid broken due to internet connection
    try:
        for key, value in tqdm(user_profile.items()):
            if key not in user_emb.keys():
                user_emb[key] = get_embedding(value)
                i = i + 1
            if i % 100 == 0:    # checkpoint
                with open("./handled/usr_profile_emb.pkl", "wb") as f:
                    pickle.dump(user_emb, f)
    except:
        pickle.dump(user_emb, open("./handled/usr_profile_emb.pkl", "wb"))
        continue
    if len(user_emb) == len(user_profile):
        break

# %%
emb_list = []
for key, value in tqdm(user_emb.items()):
    emb_list.append(value)

emb_list = np.array(emb_list)
pickle.dump(emb_list, open("./handled/usr_profile_emb.pkl", "wb"))


