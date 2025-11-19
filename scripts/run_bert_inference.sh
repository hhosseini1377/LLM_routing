python3 -m bert_routing.inference \
    --model_path ./bert_routing/finetuned_models/model_deberta_20251111-212720.pth \
    --dataset_path ./generate_dataset/datasets/MMLU/mmlu_auxiliary_and_all_with_correct_counts_n5_val.pkl \
    --model_name deberta \
    --pooling_strategy cls \
    --batch_size 32 \
    --context_window 512 \
    --output_path ./bert_routing/inference_logs