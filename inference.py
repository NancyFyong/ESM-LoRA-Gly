import torch
import random
import re
import numpy as np
import argparse
import logging
from transformers import EsmTokenizer, EsmForSequenceClassification
from peft import PeftModel

# Import your custom model classes
from model.esm_model import EsmModelClassification, EsmClassificationHead

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_fasta_file(filepath: str):
    header, sequence_parts = None, []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if sequence_parts: break
                    header = line[1:]
                else:
                    sequence_parts.append(re.sub(r'[^a-zA-Z]', '', line))
        return header, "".join(sequence_parts)
    except FileNotFoundError:
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Predict glycosylation sites using a custom-trained ESM model.")
    parser.add_argument("--fasta_file", type=str, default=None)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--type", type=str, default='N', choices=['N', 'O'])
    parser.add_argument("--base_model", type=str, default='facebook/esm2_t36_3B_UR50D')
    parser.add_argument("--lora_model", type=str, default='./lora_checkpoint')
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    set_seed(42)

    sequence_to_predict = ""
    if args.fasta_file:
        _, sequence_to_predict = read_fasta_file(args.fasta_file)
    elif args.sequence:
        sequence_to_predict = args.sequence

    if not sequence_to_predict:
        logging.error("No valid sequence provided.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- MODEL LOADING ---
    # 1. Load the base model architecture using YOUR custom class.
    #    This ensures the architecture (including the custom head) is correct.
    logging.info(f"Loading base model structure from {args.base_model} using custom EsmModelClassification class.")
    base_model = EsmModelClassification.from_pretrained(args.base_model, num_labels=2, torch_dtype=torch.float16)

    # 2. Load the LoRA weights onto the custom model structure.
    logging.info(f"Loading LoRA adapters from: {args.lora_model}")
    model = PeftModel.from_pretrained(base_model, args.lora_model)

    # 3. Merge LoRA weights and move to device.
    logging.info("Merging LoRA weights into the base model...")
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    tokenizer = EsmTokenizer.from_pretrained(args.base_model)
    logging.info("Model loaded successfully.")

    # --- PREDICTION LOGIC ---
    # 1. Identify all candidate positions to test
    candidate_residues = ['N'] if args.type == 'N' else ['S', 'T']
    candidate_positions = [i for i, aa in enumerate(sequence_to_predict) if aa in candidate_residues]
    
    # Add 1 to positions to account for the [CLS] token at the beginning
    candidate_positions_for_model = [p + 1 for p in candidate_positions]

    if not candidate_positions:
        logging.info(f"No candidate residues ({','.join(candidate_residues)}) found in the sequence.")
        return
        
    logging.info(f"Found {len(candidate_positions)} candidate sites to test.")

    # 2. Tokenize the sequence once
    inputs = tokenizer(sequence_to_predict, return_tensors="pt")
    
    all_predictions = []
    with torch.no_grad():
        # 3. Process candidates in batches for efficiency
        for i in range(0, len(candidate_positions_for_model), args.batch_size):
            batch_positions = candidate_positions_for_model[i : i + args.batch_size]
            num_in_batch = len(batch_positions)

            # 4. Create the batch
            batch_input_ids = inputs['input_ids'].repeat(num_in_batch, 1).to(device)
            batch_attention_mask = inputs['attention_mask'].repeat(num_in_batch, 1).to(device)
            batch_pos_tensor = torch.tensor(batch_positions, dtype=torch.long).to(device)
            
            # 5. Run prediction
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, pos=batch_pos_tensor)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)

    # --- DISPLAY RESULTS ---
    print("\n--- Predicted Glycosylation Sites ---")
    found_sites_count = 0
    for i, original_pos in enumerate(candidate_positions):
        prediction = all_predictions[i]
        if prediction == 1:
            found_sites_count += 1
            residue = sequence_to_predict[original_pos]
            print(f"Positive prediction -> Position: {original_pos + 1:<4}, Residue: {residue}")
            
    if found_sites_count == 0:
        logging.info("No positive glycosylation sites were predicted for the given type.")


if __name__ == "__main__":
    main()