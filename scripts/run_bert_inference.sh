python3 -m bert_routing.inference \
    --model_path ./bert_routing/finetuned_models/model_deberta_20251219-154219.pth \
    --dataset_path  ./routing_dataset/datasets/hotpotqa/hotpotqa_qwen8b_test_cleaned.pkl \
    --model_name deberta \
    --pooling_strategy cls \
    --batch_size 32 \
    --context_window 512 \
    --output_path ./bert_routing/inference_logs \