python eval.py --dataset ugphysics --model-name "/share/project/huggingface/models/QwQ-32B" --max-tokens 30000 --tensor-parallel-size 1 --data-parallel-size 8 --backend api --problem-key "question"


python eval.py --dataset ugphysics --model-name "/share/project/huggingface/models/Qwen2.5-32B-Instruct" --max-tokens 28000 --backend api

python eval.py --dataset omni-math --model-name ep-20250123001618-2sk2j --max-tokens 20000 --backend api
