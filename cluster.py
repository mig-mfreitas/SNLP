from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)

from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from tabulate import tabulate
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import sklearn

import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd


def extract_description(model_id, layer, index, json_data):

    '''
    This is how data is found in the .json file
    [
        {
            "modelId": "gemma-2-2b",
            "layer": "20-gemmascope-mlp-16k",
            "index": "4691",
            "description": " LaTeX formatting elements",
            "explanationModelName": "gpt-4o-mini",
            "typeName": "oai_token-act-pair"
        },
        {
            "modelId": "gemma-2-2b",
            "layer": "20-gemmascope-mlp-16k",
            "index": "5196",
            "description": " negations in statements",
            "explanationModelName": "gpt-4o-mini",
            "typeName": "oai_token-act-pair"
        },
        ...
    ]

    We want to extract the description given the model_id, layer, and index
    '''
    
    for data in json_data:
        if data["index"] == str(index):
            return data["description"]


def main():

    #Define release and SAE ID
    gemmascope_sae_release = "gemma-scope-2b-pt-res-canonical"
    gemmascope_sae_id = "layer_25/width_16k/canonical"

    gemma_2_2b_sae = SAE.from_pretrained(gemmascope_sae_release, gemmascope_sae_id, device="mps")[0]

    #Get decoder matrix from the SAE
    W_dec = gemma_2_2b_sae.W_dec
    W_dec = W_dec.detach().cpu().numpy()

    # First UMAP reduction (2D)

    # First UMAP reduction (2D)
    umap_2d = umap.UMAP(n_components=2, n_neighbors=15, metric="cosine", min_dist=0.05, random_state=42)
    W_umap_2d = umap_2d.fit_transform(W_dec)

    # Second UMAP reduction (10D)
    umap_10d = umap.UMAP(n_components=10, n_neighbors=15, metric="cosine", min_dist=0.1, random_state=42)
    W_umap_10d = umap_10d.fit_transform(W_dec)

    # Perform HDBSCAN clustering on 10D UMAP output
    hdb = hdbscan.HDBSCAN(min_cluster_size=5)
    clusters = hdb.fit_predict(W_umap_10d)

    #Open downloaded json 
    with open("data/gemma-2-2b_25-gemmascope-res-16k.json", "r") as f:
        json_data = json.load(f)

    #Add feature_index column to clusters
    feature_index = np.arange(len(clusters))
    clusters_expanded= np.concatenate([clusters.reshape(-1, 1), feature_index.reshape(-1, 1)], axis=1)

    #Sort clusters_expanded by cluster number
    clusters_expanded = clusters_expanded[clusters_expanded[:, 0].argsort()]

    data = []
    for i, (cluster, feature_index) in enumerate(clusters_expanded):
        #Create a triple with cluster number as "Cluster #<cluster_number>", feature index and description
        description = extract_description("gemma-2-2b", "25-gemmascope-mlp-16k", feature_index, json_data)
        data.append((f"Cluster #{cluster}", str(feature_index), description))

    df = pd.DataFrame(data, columns=["Cluster", "Feature", "Description"])
    df.set_index(["Cluster", "Feature"], inplace=True)

    #csv please
    df.to_csv("data/clusters.csv")

    #Plot 2D UMAP
    # Plot the 2D UMAP results colored by clusters
    plt.figure(figsize=(30, 30))
    scatter = plt.scatter(W_umap_2d[:, 0], W_umap_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("2D UMAP of SAE Features Colored by 10D UMAP Clusters")
    plt.show()

if __name__ == "__main__":
    main()


