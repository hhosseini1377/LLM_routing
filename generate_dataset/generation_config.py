# Generation config
class generation_config:
  model_name = "mistralai/Mistral-7B-Instruct-v0.3"
  temperature = 0.7
  top_p = 0.9
  max_tokens = 128
  max_num_sequences = 300
  dtype = "bfloat16"  # Options: "float32", "float16", "bfloat16", "auto"
  gpu_memory_utilization = 0.8
  mmlu_dataset_folder = './generate_dataset/datasets/MMLU'
  routerbench_dataset_folder = './generate_dataset/datasets/RouterBench'

class GSM8KGenerationConfig:
  model_name = "mistralai/Mistral-7B-Instruct-v0.3"
  temperature = 0.2
  top_p = 1
  max_new_tokens = 512
  max_num_sequences = 256
  dtype = "bfloat16"  # Options: "float32", "float16", "bfloat16", "auto"
  gpu_memory_utilization = 0.8
  dataset_folder = './generate_dataset/datasets/GSM8K'