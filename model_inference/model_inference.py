from utils import get_model_responses
import pandas as pd

def main():
    # Get dataset
    input_df = pd.read_csv('../data/swapped.csv').head(50) # TODO remove

    # Select models
    model_list = ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]

    # Get next token predictions
    get_model_responses(model_list,
                        input_df['sentence_text'],
                        command_prompt=None,
                        batch_size=10, #TODO remove
                        output_dir='../../output')

if __name__ == "__main__":
    main()