import torch
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
from transformers import pipeline
import os
import gc
from tqdm.notebook import tqdm


def get_token_predictions(model_name, prompts, max_new_tokens, batch_size, num_return_sequences, temperatures):
    """
    Get model next token predictions for a set of prompts and format results in a table.

    Args:
        model_name (str): HuggingFace model identifier
        prompts (pd.Series): Pandas Series of prompts
        max_length (int): Maximum length of generated text
        batch_size (int): Number of prompts per batch
        num_return_sequences (int): Number of responses per prompt
        temperatures (list): List of temperature values for generation

    Returns:
        pd.DataFrame: Results table with prompt IDs, prompts, and responses
    """
    # Initialize pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        truncation=True
    )

    # Store results
    results = []

    # Process prompts for each temperature
    for temp in temperatures:
        print(f"Processing with temperature: {temp}")
        try:
            # Generate responses for all prompts
            outputs = pipe(
                prompts.tolist(),
                batch_size=batch_size,
                temperature=temp,
                num_beams=5,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=pipe.tokenizer.pad_token_id,
                early_stopping=True,
                eos_token_id=pipe.tokenizer.eos_token_id,
            )

            # Process each response
            for prompt_id, prompt_outputs, prompt_text in zip(prompts.index, outputs, prompts):
                for seq_idx, output in enumerate(prompt_outputs):
                    generated_text = output['generated_text']
                    next_tokens = generated_text[len(prompt_text):].strip()

                    results.append({
                        'Prompt ID': prompt_id,
                        'Temperature': temp,
                        'Prompt': prompt_text,
                        'Response': next_tokens,
                        'Sequence': seq_idx + 1 if num_return_sequences > 1 else None
                    })
        
        except Exception as e:
            # Handle errors gracefully
            for prompt_id, prompt_text in zip(prompts.index, prompts):
                results.append({
                    'Prompt ID': prompt_id,
                    'Temperature': temp,
                    'Prompt': prompt_text,
                    'Response': f"Error: {str(e)}",
                    'Sequence': None
                })
        
        # Clear CUDA cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Drop Sequence column if only one sequence per prompt
    if num_return_sequences == 1:
        df.drop('Sequence', axis=1, inplace=True)

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
            'temperature': row['Temperature'],
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


def get_model_responses(model_list, prompts, command_prompt=None, max_new_tokens=50, batch_size=32, num_return_sequences=1, temperatures=[0.5, 1.0, 1.5], output_dir=None):
    # Add command prompt
    if command_prompt:
        prompts = {k: command_prompt + v for k, v in prompts.items()}
    
    # Get model responses

    output = {model_name: None for model_name in model_list}

    for model_name in model_list:
        # Get results
        results_df = get_token_predictions(
            model_name=model_name,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            num_return_sequences=num_return_sequences,
            temperatures=temperatures,
        )

        # Set filename
        filename = f"{model_name.replace('/', '-')}_completions.csv"

        # Save results to CSV
        if output_dir:
            export_completions_to_csv(results_df, num_return_sequences=num_return_sequences, output_dir=output_dir, filename=filename)

        output[model_name] = results_df

    return output