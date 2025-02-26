import torch
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
from transformers import pipeline
import os
import gc
from tqdm.notebook import tqdm


def get_model_responses(model_name, prompts, max_length=50, num_return_sequences=1):
    """
    Get model next token predictions for a set of prompts and format results in a table.

    Args:
        model_name (str): HuggingFace model identifier
        prompts (dict): Dictionary of prompts with IDs as keys
        max_length (int): Maximum length of generated text
        num_return_sequences (int): Number of responses per prompt

    Returns:
        pd.DataFrame: Results table with prompt IDs, prompts, and responses (excluding prompts)
    """
    # Initialize pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        truncation=True
    )

    # Initialize results storage
    results = []

    # Process each prompt
    for prompt_id, prompt in tqdm(prompts.items(), desc="Processing prompts"):
        try:
            # Generate responses
            outputs = pipe(
                prompt,
                num_beams=5, # beam search - maintains multiple potential output sequences (beams) during generation
                max_length=len(pipe.tokenizer.encode(prompt)) + max_length,
                num_return_sequences=num_return_sequences,
                pad_token_id=pipe.tokenizer.pad_token_id,
                early_stopping=True,
                eos_token_id=pipe.tokenizer.convert_tokens_to_ids('.'),
                temperature=0.7  # temperature - how creative the model can be
            )

            # Process each generated sequence
            for i, output in enumerate(outputs):
                generated_text = output['generated_text']

                # Remove the prompt from the generated text
                prompt_length = len(prompt)
                next_tokens = generated_text[prompt_length:].strip()

                # Add to results
                results.append({
                    'Prompt ID': prompt_id,
                    'Prompt': prompt,
                    'Response': next_tokens,  # Only include the newly generated tokens
                    'Sequence': i + 1 if num_return_sequences > 1 else None
                })

            # Clear CUDA cache after each prompt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing prompt {prompt_id}: {str(e)}")
            results.append({
                'Prompt ID': prompt_id,
                'Prompt': prompt,
                'Response': f"Error: {str(e)}",
                'Sequence': None
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Drop Sequence column if only one sequence per prompt
    if num_return_sequences == 1:
        df = df.drop('Sequence', axis=1)

    return df


def export_completions_to_csv(
    results: Dict[str, List[Tuple[str, str]]],
    num_return_sequences: int = 1,
    output_dir: str = None,
    filename: str = None
) -> str:
    """
    Export model completions to a CSV file with prompt indices.

    Args:
        results: Dictionary of results from analyze_model_completions
        output_dir: Directory path to save the CSV file
        filename: Optional custom filename

    Returns:
        str: Full path to the saved CSV file
    """
    # Prepare data for DataFrame
    rows = []
    for _, row in results.iterrows():  # Iterate through rows of DataFrame
        rows.append({
            'prompt_id': row['Prompt ID'],
            'prompt_text': row['Prompt'],
            'completion_number': row['Sequence'] if num_return_sequences > 1 else None,
            'completion_text': row['Response']
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Generate filename if not provided
    if filename is None:
        filename = "token_predictions.csv"
    elif not filename.endswith('.csv'):
        filename += '.csv'

    # Handle output path
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
    else:
        full_path = filename

    # Save to CSV
    df.to_csv(full_path, index=False)
    print(f"Results exported to: {full_path}")

    return full_path


def get_token_predictions(model_list, prompts, command_prompt=None, num_return_sequences=1, output_dir=None):
    # Add command prompt
    if command_prompt:
        prompts = {k: command_prompt + v for k, v in prompts.items()}
    
    # Get model responses

    output = {model_name: None for model_name in model_list}

    for model_name in model_list:
        # Get results
        results_df = get_model_responses(
            model_name=model_name,
            prompts=prompts,
            max_length=20,
            num_return_sequences=num_return_sequences
        )

        # Set filename
        filename = f"{model_name.replace('/', '-')}_completions.csv"

        # Save results to CSV
        if output_dir:
            export_completions_to_csv(results_df, num_return_sequences=num_return_sequences, output_dir=output_dir, filename=filename)

        output[model_name] = results_df

    return output