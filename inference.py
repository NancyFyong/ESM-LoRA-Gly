import torch
import random
import re
import numpy as np
import argparse
import logging
from transformers import EsmTokenizer, EsmForTokenClassification
from peft import PeftModel

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Predict N-linked or O-linked glycosylation sites using a fine-tuned ESM model.")
    
    parser.add_argument("--type", type=str, default='N', choices=['N', 'O'], 
                        help="Type of prediction: 'N' for N-linked or 'O' for O-linked. Default: 'N'.")
    
    parser.add_argument("--base_model", type=str, default='facebook/esm2_t36_3B_UR50D', 
                        help="Path or name of the base ESM model. Default: 'facebook/esm2_t36_3B_UR50D'.")
    
    parser.add_argument("--lora_model", type=str, default='./lora_checkpoint', 
                        help="Path to the trained LoRA model checkpoint directory. Default: './lora_checkpoint'.")
    
    parser.add_argument("--sequence", type=str, default='MNSVTVSHAPYTITYAFTVTVN', 
                        help="The amino acid sequence to predict on. Default: a short example sequence.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    set_seed(42)

    if not args.sequence:
        logging.error("Sequence cannot be empty!")
        return

    logging.info(f"Prediction Type: {args.type}")
    logging.info(f"Using sequence: {args.sequence}")

    logging.info(f"Loading base model: {args.base_model} ...")
    base_model = EsmForTokenClassification.from_pretrained(args.base_model, num_labels=2, torch_dtype=torch.float16)

    logging.info(f"Loading LoRA model from: {args.lora_model} ...")
    model = PeftModel.from_pretrained(base_model, args.lora_model)

    logging.info("Merging LoRA weights into the base model ...")
    model = model.merge_and_unload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = EsmTokenizer.from_pretrained(args.base_model)
    logging.info(f"Model successfully loaded and moved to: {device}")

    inputs = tokenizer(args.sequence, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model.eval()
    logging.info("Predicting ...")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    logging.info("Prediction finished.")

    predicted_class_ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    predictions_for_sequence = predicted_class_ids[1:-1]

    found_sites_count = 0
    if args.type == 'N':
        logging.info("Filtering for N-linked sites (N[^P][ST])...")
        sequon_pattern = r"N[^P][ST]"
        for match in re.finditer(sequon_pattern, args.sequence):
            found_sites_count += 1
            position = match.start()
            motif = match.group()
            prediction = predictions_for_sequence[position]
            print(f"Position: {position + 1:<4}, Motif: {motif}, Prediction: {prediction}")
    
    elif args.type == 'O':
        logging.info("Filtering for O-linked sites (S or T)...")
        for i, amino_acid in enumerate(args.sequence):
            if amino_acid in ['S', 'T']:
                found_sites_count += 1
                position = i
                prediction = predictions_for_sequence[position]
                print(f"Position: {position + 1:<4}, Residue: {amino_acid}, Prediction: {prediction}")

    if found_sites_count == 0:
        logging.warning("No potential sites found for the protein")

if __name__ == "__main__":
    main()