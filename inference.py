import torch
import random
import re
import numpy as np # 建议导入numpy
from transformers import EsmTokenizer, EsmForTokenClassification
from peft import PeftModel
from logging import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def set_seed(seed: int = 42):
    """
    固定所有随机种子以确保结果的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    
    # 配置PyTorch使用确定性的CUDA算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"所有随机种子已固定为: {seed}")
    print(f"PyTorch确定性算法已开启。")

set_seed(42) 


base_model_name = "facebook/esm2_t36_3B_UR50D"
lora_model_path = "/checkpoint"

sequence = ""



# --- 加载模型和分词器 (不变) ---
# ... (加载和合并模型的代码与之前完全相同)
print(f"1. 正在加载基础模型: {base_model_name}")
base_model = EsmForTokenClassification.from_pretrained(base_model_name, num_labels=2, torch_dtype=torch.float16)
print(f"2. 正在从 '{lora_model_path}' 加载并附加LoRA权重...")
model = PeftModel.from_pretrained(base_model, lora_model_path)
print("3. 正在合并LoRA权重以提高推理速度...")
model = model.merge_and_unload()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"模型已移动到: {device}\n")
tokenizer = EsmTokenizer.from_pretrained(base_model_name)


# --- 预测 ---
inputs = tokenizer(sequence, return_tensors="pt")
inputs = {key: val.to(device) for key, val in inputs.items()}

# ####################################################################
#  ↓↓↓ 步骤2: 确保模型处于评估模式 (您的代码已正确执行) ↓↓↓
# ####################################################################
model.eval() 

logging.info("正在进行预测 (使用您微调后的LoRA模型)...")
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
print("预测完成！\n")

predicted_class_ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]
predictions_for_sequence = predicted_class_ids[1:-1]
sequon_pattern = r"N[^P][ST]"

found_sites_count = 0
for match in re.finditer(sequon_pattern, sequence):
    found_sites_count += 1
    position = match.start()
    motif = match.group()
    prediction = predictions_for_sequence[position]
    print(f"发现位点: 位置 {position + 1:<4}, 基序: {motif}, 预测类别: {prediction}")
if found_sites_count == 0:
    print("在输入序列中未发现任何 'N-X-[S/T] (X≠P)' 模式的位点。")