import json
import os

import pandas as pd
import torch
from pydpp.dpp import DPP
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


###FUNCTIONS

    ###EMBEDDINGS EXTRACTION
def get_embeddings(df, model):
    sentences = df['text'].to_list()
    embeddings = model.encode(sentences)
    return embeddings

    ###DPP CALCULATION
def get_dpp_points(dpp, num_samples, df):
    dpp.compute_kernel(kernel_type = 'cos-sim', sigma= 0.4)
    samples_n = dpp.sample_k(num_samples) # These will return the indices 
    # Be very careful you have to use the index column not then index in itself
    df_sampled = df.iloc[samples_n]
    return df_sampled   


    ###SAMPLING SHOTS
def generate_shots(dpp_hp, dpp_non_hp, num_samples, df, df_fake,df_true):
    """
    Generate 'shot' lists from provided DataFrames and DPP points.
    """
    #df_samples_right = get_dpp_points(dpp_right, num_samples=5, df=df_right)

    # Create the 'shot' column

    df_samples_fake["shot"] = df_samples_fake["text"] + ' , '+ "'"+df_samples_fake["label"].astype(str) + "'"
    F_list = df_samples_fake['shot'].to_list() #non hyperpartisan shots

    df_samples_true["shot"] = df_samples_true["text"] + ' , '+"'"+df_samples_true["label"].astype(str) + "'"
    T_list = df_samples_true['shot'].to_list()

    # Convert to lists
    return F_list, T_list

DATA = "FakeNewsCorpusSpanish"
###VARIABLES
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer("embaas/sentence-transformers-multilingual-e5-large")

###DATASET PROCESSING
df = pd.read_csv(os.path.join(os.path.expanduser('~/datasets/subdatasets'), DATA+'.csv'))


df=df.rename(columns = {'Unnamed: 0': 'index'})
#df['label'] = df.label.map({'left':0, 'center':1, 'right':2})
df=df.rename(columns = {'Category': 'label'})
df=df.rename(columns = {'Text': 'text'})

#df['label'] = df.label.map({'real':0, 'fake':1})
df = df.sample(frac = 1).reset_index(drop = True)
df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42) # We do this for sanity check
df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

###SENTENCE TRANSFORMERS Calculation

df_fake= df_train[df_train['label'] == 1].reset_index(drop = True)
df_true = df_train[df_train['label'] == 0].reset_index(drop = True)


embeddings_fake = get_embeddings(df_fake, model)
embeddings_true = get_embeddings(df_true, model)
print("Embedding calculated:", df_fake.shape, df_true.shape)



dpp_fake = DPP(embeddings_fake) 
dpp_true = DPP(embeddings_true)


print("DPP calcutalted. Now passing to sampling DPP.")
runs = {}
for k in range(5):
    # Generate the shots
    df_samples_fake = get_dpp_points(dpp_fake, num_samples=5, df=df_fake)
    df_samples_true = get_dpp_points(dpp_true, num_samples=5, df=df_true)

    F_list, T_list = generate_shots(dpp_fake, dpp_true, num_samples=5, df_F=df_fake, df_T = df_true)

    dic_F = {f"h_shot_run_{i+1}": el for i, el in enumerate(F_list)}
    dic_T = {f"h_shot_run_{i+1}": el for i, el in enumerate(T_list)}


    runs[f"run_n_{k+1}"] = {'dic_F': dic_F, 'dic_T':dic_T}

print("Going to save samples.")

# Save the runs dictionary to a JSON file
with open(f"{DATA}_runs_output.json", "w") as json_file:
    json.dump(runs, json_file, indent=4)

