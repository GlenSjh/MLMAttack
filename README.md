Code and models for the paper "Towards a Unified Framework for Imperceptible Textual Attacks"

Train clean model as well as finetuning MLM

```python train.py --dataset sst2 --model_name bert-base-uncased --epochs 3 --lr_model 2e-5 --batch 32 --type clean --output_model_path saved_models/clean/sst2/bert-base-uncased.pkl```

Perform targeted adversarial attack to the clean model

```python train.py --dataset sst2 --model_name bert-base-uncased --batch 32 --type adv --weight_lambda 0.25 --target_label 1 --clean_model_path saved_models/clean/sst2/bert-base-uncased.pkl```

Perform backdoored adversarial attack to the clean model

```python train.py --dataset sst2 --model_name bert-base-uncased --batch 32 --type backdoor --weight_lambda 0.7 --clean_model_path saved_models/clean/sst2/bert-base-uncased.pkl```




