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