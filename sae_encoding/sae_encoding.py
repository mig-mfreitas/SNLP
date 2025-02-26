from utils import get_sae_embeddings
import pandas as pd

def main():
    # Select models
    model_list = ["Qwen/Qwen2.5-0.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]

    for model_name in model_list:
        # Get saved completions
        input_df = pd.read_csv(f'../../output/{model_name.replace("/", "-")}_completions.csv')
        input_list = input_df["completion_text"].tolist()

        # Get embeddings for each model's completions
        get_sae_embeddings(input_list,
                           gemma_scope_sae_release="gemma-scope-2b-pt-res-canonical",
                           gemma_scope_sae_id="layer_25/width_16k/canonical")

if __name__ == "__main__":
    main()