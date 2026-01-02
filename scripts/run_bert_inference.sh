python3 -m bert_routing.inference \
    --model_path ./bert_routing/finetuned_models/model_deberta_20260101-163459.pth \
    --dataset_path  ./routing_dataset/datasets/lmsys_chat1m/lmsys_chat1m_test.pkl \
    --model_name deberta \
    --pooling_strategy cls \
    --batch_size 32 \
    --context_window 256 \
    --output_path ./bert_routing/inference_logs \