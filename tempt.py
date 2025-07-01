from transformers import AutoModelForCausalLM, AutoTokenizer
from config import generator_config
model_id = "openchat/openchat-3.5-1210"
save_path = "./models/openchat-3.5-1210-local"

# Load and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_path)

# Load and save model (ensure weights are downloaded)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.save_pretrained(save_path)
