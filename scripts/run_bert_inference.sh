python3 -m bert_routing.inference \
    --model_path ./bert_routing/finetuned_models/model_deberta_20251218-122746.pth \
    --dataset_path  ./routing_dataset/datasets/final_splits/mmlu_all_pro_gsm8k_qwen8b_test.pkl \
    --model_name deberta \
    --pooling_strategy cls \
    --batch_size 32 \
    --context_window 512 \
    --output_path ./bert_routing/inference_logs \