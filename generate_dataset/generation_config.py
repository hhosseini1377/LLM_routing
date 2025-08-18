# Generation config
class generation_config:
  model_name = "mistralai/Mistral-7B-Instruct-v0.1"
  temperature = 0.3
  top_p = 0.95
  max_tokens = 128
  max_num_sequences = 128
  mmlu_dataset_folder = './generate_dataset/datasets/MMLU'
  routerbench_dataset_folder = './generate_dataset/datasets/RouterBench'