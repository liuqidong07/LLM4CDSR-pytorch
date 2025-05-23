# LLM4CDSR: Large Language Models Can Be a Good Bridge for Cross-domain Sequential Recommendation

This is the implementation of the paper "LLM4CDSR: Large Language Models Can Be a Good Bridge for Cross-domain Sequential Recommendation".

## Configure the environment

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: RTX 3090 24GB
  - Cuda: 12.0
  - Driver version: 525.105.17
  - CPU: AMD EPYC 7543 32-Core
- Software:
  - Python: 3.9.16
  - Pytorch: 2.3.1

You can pip install the `requirements.txt` to configure the environment.

## Preprocess the dataset

You can preprocess the dataset and get the LLMs embedding according to the following steps:

1. The raw dataset downloaded from website should be put into `/data/<amazon/elec/douban>/raw/`. The Douban dataset can be obtained from this [Download Link](https://www.researchgate.net/publication/350793434_Douban_dataset_ratings_item_details_user_profiles_tags_and_reviews). The amazon and elec datasets can be obtained from this [Download Link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).
2. Conduct the preprocessing code `data/<amazon/elec/douban>/handle.ipynb` to get the data for cross-domain sequential recommendation. After the procedure, you will get the id file  `/data/<amazon/elec/douban>/hdanled/id_map.json` and the interaction file  `/data/<amazon/elec/douban>/handled/<cloth-sport/elec-phone/book-movie>.pkl`.
3. To get the LLMs embedding for cross-domain items, please run the jupyter notebooks `/data/<amazon/elec/douban>/item_prompt.ipynb`. After the running, you will get the LLMs item embedding file `/data/<<amazon/elec/douban>/handled/itm_emb_np.pkl`. Then, run the `/data/<amazon/elec/douban>/pca.ipynb` to get their dimension-reduced unified LLM embedding file `itm_emb_np_all.pkl` and local embedding `itm_emb_np_<A/B>_pca128.pkl`.
4. To get the user's comprehensive profiles and correpsonding LLM embeddings, run the python file `/data/<amazon/elec/douban>/user_profile.py`, and then you can get `usr_profile_emb.pkl`.

In conclusion, the prerequisite files to run the code are as follows: `<cloth-sport/elec-phone/book-movie>.pkl`, `itm_emb_np_all.pkl`, `itm_emb_np_<A/B>_pca128.pkl`, `usr_profile_emb.pkl` and `id_map.json`.

⭐️ To ease the reproducibility of our paper, we also upload all preprocessed files to this [link](https://drive.google.com/file/d/1QWY4N3LCWHNkFLgIYVorFR-Ippkrjoee/view?usp=sharing).

## Run and test

1. You can reproduce all LLM4CDSR experiments by running the bash as follows:

```
bash experiments/<amazon/elec/douban>/llm4cdsr.bash
```

2. The log and results will be saved in the folder `log/`. The checkpoint will be saved in the folder `saved/`.
