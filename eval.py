import json
import jsonlines
import os
import re
import time
import argparse

from pathlib import Path
from datasets import load_dataset
from metric import AIMEEvaluator, OminiMathEvaluator
from tqdm import tqdm
from template_api import TemplateAPI

try:
    from vllm import LLM, SamplingParams
except:
    print("Failed to import VLLM")

try:
    import sglang as sgl
except:
    print("Failed to import sglang")

PROMPT_TEMPLATES = {
    "qwen-boxed": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n",
    "qwen25-math-cot": "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
}

HF_DATASETS = [
    "KbsdJames/Omni-MATH",
]

DATASET_EVALUATORS = {
    "aime": AIMEEvaluator(),
    "KbsdJames/Omni-MATH": OminiMathEvaluator("KbsdJames/Omni-MATH"),
}

def get_evaluator(dataset_name):
    """Get evaluator for dataset"""
    if dataset_name in DATASET_EVALUATORS:
        return DATASET_EVALUATORS[dataset_name]
    
    for name, evaluator in DATASET_EVALUATORS.items():
        if name in dataset_name or dataset_name in name:
            return evaluator
    
    raise ValueError(f"No evaluator found for dataset {dataset_name}")


def process_deepseek_prediction(prediction):
    match = re.search(r"</think>\n\n(.*)", prediction, re.DOTALL)
    return match.group(1) if match else prediction

def load_model_by_backend(backend, model_name, args):

    if backend == 'vllm':
        model_path = model_name
        model_args = {
            "model": model_path,
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "trust_remote_code": args.trust_remote_code,
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "seed": int(args.seed),
            "max_model_len": int(args.max_tokens),
        }

        start_time = time.time()

        llm = LLM(**model_args)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        
        return llm, sampling_params
    
    elif backend == "sglang":
        model_path = model_name
        model_args = {
            "model_path": model_path,
            "mem_fraction_static": float(args.gpu_memory_utilization),
            "tp_size": int(args.tensor_parallel_size),
            "dp_size": int(args.data_parallel_size),
        }

        start_time = time.time()
        llm = sgl.Engine(**model_args)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

        sampling_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens
        }
        
        return llm, sampling_params
    elif backend == "api":

        model = TemplateAPI(
            model=model_name,
            api_key=None,
            base_url="http://127.0.0.1:8000/v1/chat/completions",
            num_concurrent=4,
        )
        sampling_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens
        }
        return model, sampling_params
    else:
        raise ValueError(f"Invalid backend: {backend}")



def promptify(prompt, prompt_template, tokenizer):
    if not prompt_template and not tokenizer:
        return [{"role": "user", "content": prompt}]
    
    if tokenizer.chat_template != None:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    
    return PROMPT_TEMPLATES[prompt_template].format(input=prompt)
    

def process_jsonl_file(dataset, model_name, args, output_path=None, problem_key='problem'):
    # Load model
    llm, sampling_params = load_model_by_backend(args.backend, model_name=model_name, args=args)

    # Process prompts
    tokenizer = llm.get_tokenizer() if args.backend != "api" else None
    prompt_template = args.prompt_template if args.backend != "api" else None
    prompts = [promptify(sample[problem_key], prompt_template, tokenizer) for sample in dataset]

    # Run inference using a unified function call
    start_time = time.time()
    if args.backend != "api":
        outputs = llm.generate(prompts, sampling_params)
    else:
        outputs = llm.generate_until(prompts, gen_kwargs=sampling_params)
    print(f"Inference for {len(prompts)} samples took {time.time() - start_time:.2f} seconds")

    # Process outputs
    predictions = [
        output.outputs[0].text if isinstance(output, dict) else output
        for output in outputs
    ]
    processed_predictions = [process_deepseek_prediction(pred) for pred in predictions]

    # Combine results
    processed_data = [{**sample, "prediction": pred} for sample, pred in zip(dataset, processed_predictions)]
    
    # Save results
    if output_path:
        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(processed_data)
        print(f"Saved {len(processed_data)} samples to {output_path}")
    
    return processed_data

    
def load_task_dataset(task_name):
    """Load dataset with attached evaluators"""
    dataset = []

    def add_evaluator(items, task_id):
        for item in items:
            item["task_name"] = task_id
        return items
    
    data_dir = Path("data")

    if task_name == "all":
        if data_dir.exists():
            for task_dir in data_dir.iterdir():
                if task_dir.is_dir():
                    task_dataset_path = task_dir / "test.jsonl"
                    with jsonlines.open(task_dataset_path, 'r') as reader:
                        task_data = add_evaluator(list(reader), task_dir.name)
                        dataset.extend(task_data)

        for hf_dataset_name in HF_DATASETS:
            try:
                hf_dataset = load_dataset(hf_dataset_name)
                task_data = list(hf_dataset["test"]) if "test" in hf_dataset else list(hf_dataset["train"])
                dataset.extend(add_evaluator(task_data, hf_dataset_name))
            except Exception as e:
                print(f"Warning: Failed to load {hf_dataset_name}: {str(e)}")

    else:
        data_path = data_dir / task_name
        if os.path.exists(data_path):
            if data_path.is_file() and data_path.suffix == '.jsonl':
                with jsonlines.open(data_path, 'r') as reader:
                    dataset = add_evaluator(list(reader), data_path.stem)
            elif data_path.is_dir():
                task_dataset_path = data_path / "test.jsonl"
                if task_dataset_path.exists():
                    with jsonlines.open(task_dataset_path, 'r') as reader:
                        dataset = add_evaluator(list(reader), data_path.name)
        else:
            try:
                hf_dataset = load_dataset(task_name)
                task_data = list(hf_dataset["test"]) if "test" in hf_dataset else list(hf_dataset["train"])
                dataset = add_evaluator(task_data, task_name)
            except Exception as e:
                raise ValueError(f"Failed to load {task_name}: {str(e)}")

    if not dataset:
        raise ValueError(f"Dataset '{task_name}' not found")
        
    return dataset


def analyze_results(data, answer_key='answer'):
    """Analyze dataset-level results after individual sample evaluation"""
    # Group samples by task
    task_data = {}
    for sample in data:
        task = sample['task_name']
        if task not in task_data:
            task_data[task] = []
        task_data[task].append(sample)
    
    task_results = {}
    
    for task, samples in task_data.items():
        evaluator = get_evaluator(task)
        
        # Sample-level evaluation first
        evaluation_results = []
        for sample in tqdm(samples, desc=f"Evaluating {task}"):
            is_correct = evaluator.evaluate(sample['prediction'], sample[answer_key])
            sample['correct'] = is_correct
            evaluation_results.append(is_correct)
        
        breakpoint()
        # Dataset-level analysis
        # Use evaluator's analyze_results if available, otherwise use calculate_accuracy_stats
        if evaluator.get_name() == "ominimath":
            task_results[task] = evaluator.analyze_results(samples)
        else:
            task_stats = {
                task: {
                    'accuracy': sum(evaluation_results) / len(evaluation_results),
                    'correct': sum(evaluation_results),
                    'total': len(evaluation_results)
                }
            }
            task_results[task] = task_stats[task]
    
    return task_results

def main():
    parser = argparse.ArgumentParser(description="Process datasets with fast inference.")
    parser.add_argument("--dataset", required=True, help="Name or path of the dataset to load.")
    parser.add_argument("--split", default="test", help="Split to use (if dataset has multiple splits).")
    parser.add_argument("--model-name", required=True, help="Name of the model to use for inference.")
    parser.add_argument("--problem-key", default="problem", help="Key for the problem/prompt in the dataset.")
    parser.add_argument("--answer-key", default="answer", help="Key for the answer/ground truth in the dataset.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for generation (default: 20000)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for nucleus sampling (default: 1.0)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization (default: 0.9)")
    parser.add_argument("--trust-remote-code", type=bool, default=False, help="Whether to trust remote code (default: False)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument("--data-parallel-size", type=int, default=1, help="Data parallel size for sglang (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation (default: 42)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for inference (default: 100)")
    parser.add_argument("--prompt-template", default="qwen25-math-cot", help="Key for prompt template.")
    parser.add_argument("--backend", default="vllm", help="Backend to use for inference (default: vllm)")

    args = parser.parse_args()

    # Load dataset
    original_data = load_task_dataset(args.dataset)

    # Define output path (assuming args is available in the scope)
    output_path = Path("eval_output") / f"processed_{os.path.basename(args.dataset).split('.')[0]}_{args.split or 'all'}_{args.model_name.split('/')[-1]}_{args.max_tokens}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process data without converting to list
    processed_data = process_jsonl_file(
        original_data, args.model_name, args, output_path, args.problem_key)

    # Analyze results
    eval_result = analyze_results(processed_data, args.answer_key)

    if eval_result:
        # Print accuracy for each task to console
        for task_name, stats in eval_result.items():
            accuracy_str = f"Exact match accuracy for {task_name}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})"
            print(accuracy_str)
        
        # Prepare data for JSON output
        eval_output_path = Path("eval_results.json")
        eval_data = {
            "dataset": args.dataset,
            "model": args.model_name,
            "results": eval_result,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }   
        
        # Write evaluation results to JSON Lines file
        eval_json = json.dumps(eval_data, indent=4)
        with open(eval_output_path, 'a', encoding='utf-8') as f:
            f.write(eval_json + '\n')
        
        print(f"Evaluation result written to {eval_output_path}")

if __name__ == "__main__":
    main()