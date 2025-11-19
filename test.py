import pickle
import numpy as np

from router_system.compute_flops import compute_flops_for_different_thresholds_hierarchical_routing, analyze_output_token_lengths, compute_reliability_for_different_thresholds_hierarchical_routing


# Example: Analyze prompts from a pickle file
# Replace with your actual file path
file_path = './generate_dataset/datasets/MMLU/mmlu_auxiliary_and_all_with_correct_counts_n5_val.pkl'  # Example path
cpx_prob_dir = 'cpx_model/inference_logs/probabilities_20251116-214238.pkl'
bert_prob_dir = 'bert_routing/inference_logs/probabilities_20251117-151544.pkl'
reliability_results = compute_reliability_for_different_thresholds_hierarchical_routing(cpx_prob_dir, bert_prob_dir, 7, 14)
print(reliability_results)