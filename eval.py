import json
import jsonlines
import os
import re
import time
import argparse

from pathlib import Path
from datasets import load_dataset
from metric import AIMEEvaluator, OminiMathEvaluator, UgPhysicsEvaluator
from tqdm import tqdm
from template_api import CachingLM, TemplateAPI
from token_soup import get_token

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
    "omni-math": OminiMathEvaluator(),
    "ugphysics": UgPhysicsEvaluator()
}

def get_evaluator(dataset_name):
    """Get evaluator for dataset"""
    if dataset_name in DATASET_EVALUATORS:
        return DATASET_EVALUATORS[dataset_name]
    
    for name, evaluator in DATASET_EVALUATORS.items():
        if name in dataset_name or dataset_name in name:
            return evaluator
    
    raise ValueError(f"No evaluator found for dataset {dataset_name}")


# def process_deepseek_prediction(prediction):
#     match = re.search(r"</think>\n\n(.*)", prediction, re.DOTALL)
#     return match.group(1) if match else prediction

def process_deepseek_prediction(prediction):
    think_match = re.search(r"(.*?)</think>\n\n(.*)", prediction, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        final_answer = think_match.group(2).strip()
        return {
            "thinking": think_content,
            "answer": final_answer,
            "full_response": prediction
        }
    return {
        "thinking": "",
        "answer": prediction,
        "full_response": prediction
    }

def load_model_by_backend(args):

    if args.backend == 'vllm':
        model_path = args.model_name
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
    
    elif args.backend == "sglang":
        model_path = args.model_name
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
    
    elif args.backend == "api":

        api_key = None
        model = args.model_name

        if "share" not in args.model_name:
            model_config, model_name = get_token(
                args.model_name, 
                use_fuzzy=True, 
                threshold=80
            )
            model = model_name
            api_key = model_config["api_key"]
            base_url = model_config["base_url"]
            num_concurrent = 36
        else:
            if args.use_chat_completions:
                base_url = "http://127.0.0.1:8000/v1/chat/completions"
            else:
                base_url = "http://127.0.0.1:8000/v1/completions"

            num_concurrent = 12

        print(f"Using API model with base URL: {base_url}")
        
        
        model = TemplateAPI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            num_concurrent=num_concurrent,
        )

        # Set up caching if requested
        if args.use_cache:
            print(f"Using cache: {args.use_cache}")
            model = CachingLM(model, args.use_cache)

        sampling_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "repetition_penalty": args.repetition_penalty
        }
        return model, sampling_params
    else:
        raise ValueError(f"Invalid backend: {args.backend}")



def promptify(prompt, prompt_template, tokenizer, use_chat_completions=False, use_system_prompt=True):
    if not use_chat_completions:
        return prompt

    # Convert prompt to messages list if needed and build base messages
    messages = prompt if isinstance(prompt, list) else []
    if not messages:
        messages = []
        if use_system_prompt:
            messages.append({
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}."
            })
        messages.append({"role": "user", "content": prompt})
    
    # Handle case with no template/tokenizer
    if not prompt_template and not tokenizer:
        return messages

    # Use tokenizer chat template if available
    if tokenizer and tokenizer.chat_template is not None:
        # Directly use the already constructed messages list
        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Fall back to prompt template
    if prompt_template not in PROMPT_TEMPLATES:
        raise ValueError(f"Invalid prompt template: {prompt_template}")
    
    return PROMPT_TEMPLATES[prompt_template].format(input=prompt)
    

# def process_conversations(data, model_name, model_config, concurrency=2, use_cache=None):
#     """Process all conversations with the model"""
    
#     max_turns = 2
    
#     # Initialize conversation states
#     messages_list = [[] for _ in data]
    
#     prompt_list = [
#         ""
#     ]
#     # Process each turn
#     for turn_idx in range(max_turns):
#         # Collect samples for this turn
#         batch_indices = []
#         batch_prompts = []
        
#         for i, sample in enumerate(data):
#             problem = sample.get('problem', sample.get("question"))
#             if turn_idx < max_turns:
#                 # Add user message
#                 messages_list[i].append({"role": "user", "content": turns[turn_idx]})
#                 batch_indices.append(i)
#                 batch_prompts.append((f"sample_{i}_turn_{turn_idx}", messages_list[i][:]))
        
#         # Generate responses
#         if batch_prompts:
#             print(f"Turn {turn_idx}: Processing {len(batch_prompts)} conversations")
#             batch_predictions = model.generate_until(
#                 batch_prompts, 
#                 gen_kwargs={"max_gen_toks": 20000}
#             )
            
#             # Add assistant responses
#             for idx, prediction in zip(batch_indices, batch_predictions):
                
#                 messages_list[idx].append({"role": "assistant", "content": process_predictions(prediction)})
    
#     # Prepare results
#     results = []
#     for i, sample in enumerate(data):
#         # Extract assistant responses
#         predictions = [msg["content"] for msg in messages_list[i] if msg["role"] == "assistant"]
#         results.append({**sample, 'prediction': predictions})
    
#     return results

def process_jsonl_file(dataset, args):
    # Load model
    llm, sampling_params = load_model_by_backend(args=args)

    # Process prompts
    tokenizer = llm.get_tokenizer() if args.backend != "api" else None
    prompt_template = args.prompt_template if args.backend != "api" else None
    problem_key = get_problem_key(dataset)
    prompts = [promptify(sample[problem_key], prompt_template, tokenizer, use_chat_completions=args.use_chat_completions, use_system_prompt=args.use_system_prompt) for sample in dataset]

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

    processed_data = [{
        **sample, 
        "prediction": pred["answer"],
        "thinking": pred["thinking"],
        # "full_response": pred["full_response"],
        "input": str(processed_prompt)
    } for sample, pred, processed_prompt in zip(dataset, processed_predictions, prompts)]
        
    return processed_data

    
def load_task_dataset(task_name):
    """Load dataset with attached evaluators"""
    dataset = []

    def add_evaluator(items, task_id):
        for id, item in enumerate(items):
            item['id'] = id
            item["task_name"] = task_id
        return items
    
    def normalize_keys(items):
        for item in items:
            for key in ['problem', 'question', 'query']:
                if key in item:
                    item['problem'] = item.pop(key)
                    break
            for key in ['answer', 'answers']:
                if key in item:
                    item['answer'] = item.pop(key)
                    break
        return items
    
    data_dir = Path("data")

    if task_name == "all":
        if data_dir.exists():
            for task_dir in data_dir.iterdir():
                if task_dir.is_dir():
                    task_dataset_path = task_dir / "test.jsonl"
                    with jsonlines.open(task_dataset_path, 'r') as reader:
                        task_data = normalize_keys(list(reader))
                        task_data = add_evaluator(task_data, task_dir.name)
                        dataset.extend(task_data)

        for hf_dataset_name in HF_DATASETS:
            try:
                hf_dataset = load_dataset(hf_dataset_name)
                task_data = list(hf_dataset["test"]) if "test" in hf_dataset else list(hf_dataset["train"])
                task_data = normalize_keys(task_data)
                dataset.extend(add_evaluator(task_data, hf_dataset_name))
            except Exception as e:
                print(f"Warning: Failed to load {hf_dataset_name}: {str(e)}")

    else:
        data_path = data_dir / task_name
        if os.path.exists(data_path):
            if data_path.is_file() and data_path.suffix == '.jsonl':
                with jsonlines.open(data_path, 'r') as reader:
                    dataset = normalize_keys(list(reader))
                    dataset = add_evaluator(dataset, data_path.stem)
            elif data_path.is_dir():
                task_dataset_path = data_path / "test.jsonl"
                if task_dataset_path.exists():
                    with jsonlines.open(task_dataset_path, 'r') as reader:
                        dataset = normalize_keys(list(reader))
                        dataset = add_evaluator(dataset, data_path.name)
        else:
            try:
                hf_dataset = load_dataset(task_name)
                task_data = list(hf_dataset["test"]) if "test" in hf_dataset else list(hf_dataset["train"])
                task_data = normalize_keys(task_data)
                dataset = add_evaluator(task_data, task_name)
            except Exception as e:
                raise ValueError(f"Failed to load {task_name}: {str(e)}")

    if not dataset:
        raise ValueError(f"Dataset '{task_name}' not found")
        
    return dataset


def analyze_results(data, llm_as_judge=False, answer_key='answer'):
    """Analyze dataset-level results after individual sample evaluation"""
    # Group samples by task
    task_data = {}
    for sample in data:
        task = sample['task_name']
        if task not in task_data:
            task_data[task] = []
        task_data[task].append(sample)
    
    task_results = {}
    processed_data = []
    
    for task, samples in task_data.items():
        evaluator = get_evaluator(task)
        
        # Sample-level evaluation first
        evaluation_results = []
        
        # Check if evaluator supports batch evaluation (currently only omni-math)
        if hasattr(evaluator, 'evaluate_by_judge') and llm_as_judge:
            # Prepare batch data
            predictions = [sample['prediction'] for sample in samples]
            answers = [sample[answer_key] for sample in samples]
            questions = [sample.get('question', None) for sample in samples]
            
            # Batch evaluate
            batch_results = evaluator.evaluate_by_judge(predictions, answers, questions)
            
            # Process results
            for sample, is_correct in tqdm(zip(samples, batch_results), desc=f"Evaluating {task}", total=len(samples)):
                sample['correct'] = is_correct
                evaluation_results.append(is_correct)
                processed_data.append(sample)
        else:
            # Regular sample-by-sample evaluation
            for sample in tqdm(samples, desc=f"Evaluating {task}"):
                is_correct = evaluator.evaluate(sample['prediction'], sample[answer_key])
                sample['correct'] = is_correct
                evaluation_results.append(is_correct)
                processed_data.append(sample)
        
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
    
    return task_results, processed_data


def get_output_path(args):
    # Create base directory if it doesn't exist
    output_dir = Path("eval_output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Map common delimiters to their string names
    delimiter_names = {
        ".": "period",
        ",": "comma",
        " ": "space",
        ";": "semicolon",
        ":": "colon",
        "-": "dash",
        "_": "underscore",
        "\\n": "newline",
        "\\t": "tab"
    }

    # Extract model name without path
    model_short_name = args.model_name.split('/')[-1]
    
    # Build descriptive filename with parameters
    params = [
        f"model_{model_short_name}",
        f"temp_{args.temperature}",
        f"rep_{args.repetition_penalty}",
        f"chat_{args.use_chat_completions}"
    ]
    
    # Add debug dataset info if available
    if hasattr(args, 'load_debug_dataset') and args.load_debug_dataset:
        # Convert delimiter to string name if it exists in our mapping
        delimiter_name = delimiter_names.get(args.delimiter.strip(), args.delimiter.strip())
        params.append(f"debug_{args.load_debug_dataset.split('/')[-1].removesuffix('.jsonl')}_{delimiter_name}")
        
    
    # Join with underscores for filename
    filename = "_".join(params) + ".jsonl"
    
    return output_dir / filename

def main():
    parser = argparse.ArgumentParser(description="Process datasets with fast inference.")
    parser.add_argument("--dataset", required=True, help="Name or path of the dataset to load.")
    parser.add_argument("--split", default="test", help="Split to use (if dataset has multiple splits).")
    parser.add_argument("--model-name", required=True, help="Name of the model to use for inference.")
    parser.add_argument("--answer-key", default="answer", help="Key for the answer/ground truth in the dataset.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for generation (default: 8192)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling (default: 0.6)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for nucleus sampling (default: 1.0)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty for generation (default: 1.0)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization (default: 0.9)")
    parser.add_argument("--trust-remote-code", type=bool, default=False, help="Whether to trust remote code (default: False)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument("--data-parallel-size", type=int, default=1, help="Data parallel size for sglang (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation (default: 42)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for inference (default: 1024)")
    parser.add_argument("--prompt-template", default="qwen25-math-cot", help="Key for prompt template.")
    parser.add_argument("--backend", default="vllm", help="Backend to use for inference (default: vllm)")
    parser.add_argument("--use-cache", action="store_true", help="Whether to use cache for inference")
    parser.add_argument("--use-chat-completions", type=bool, default=True, help="Use chat completions mode for API models.")
    parser.add_argument("--use-system-prompt", type=bool, default=True, help="Whether to use the system prompt.")
    parser.add_argument("--llm-as-judge", type=bool, default=False, help="Whether to use the LLM as judge.")
    parser.add_argument("--load-debug-dataset", type=str, help="Load the dataset for debugging")
    parser.add_argument("--delimiter", type=str, default=". ")

    args = parser.parse_args()

    # Load dataset
    if args.load_debug_dataset:
        print(f"[debug]: args.delimiter={args.delimiter}")
        with jsonlines.open(args.load_debug_dataset, 'r') as reader:
            data = list(reader)
            original_data = []
            for id, item in enumerate(data):

                query = eval(item['input'])
                thinking_content = item['thinking']
                prediction_content = item['prediction']
                answer = item['answer']
                task_name = item['task_name']
                question = item['question']

                content_message = query[0]['content'] + " "
                for _id, sub_sentence in enumerate(thinking_content.split(args.delimiter)):
                    content_message += sub_sentence + args.delimiter
                    sample = {
                        "id": id,
                        "input": [{"role": "user", "content": content_message}],
                        "question": question,
                        "task_name": task_name,
                        "answer": answer,
                        "label": f"thinking_split:{_id}"
                    }
                    original_data.append(sample)

                for _id, sub_sentence in enumerate(prediction_content.split(args.delimiter)):
                    content_message += sub_sentence + args.delimiter
                    sample = {
                        "id": id,
                        "input": [{"role": "user", "content": content_message}],
                        "question": question,
                        "task_name": task_name,
                        "answer": answer,
                        "label": f"prediction_split:{_id}"
                    }
                    original_data.append(sample)
                # breakpoint()
    else:
        original_data = load_task_dataset(args.dataset)

    # Define output path (assuming args is available in the scope)
    output_path = get_output_path(args)
    print(f"output_path={output_path}")
    # Process data without converting to list
    processed_data = process_jsonl_file(original_data, args)


    # Analyze results
    eval_result, processed_data_with_metric = analyze_results(processed_data, args.llm_as_judge, args.answer_key)

    if eval_result:
        # Print accuracy for each task to console
        for task_name, stats in eval_result.items():
            accuracy_str = f"Exact match accuracy for {task_name}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})"
            print(accuracy_str)
        
        # Prepare data for JSON output
        eval_output_path = Path("eval_results.json")
        eval_data = {
            "dataset": args.dataset if not args.load_debug_dataset else args.load_debug_dataset,
            "model": args.model_name,
            "results": eval_result,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "use_chat_completions": args.use_chat_completions,
            "timestamp": time.time(),
        }   
        
        # Write evaluation results to JSON Lines file
        eval_json = json.dumps(eval_data, indent=4)
        with open(eval_output_path, 'a', encoding='utf-8') as f:
            f.write(eval_json + '\n')
        
        print(f"Evaluation result written to {eval_output_path}")

        # Save results
    if output_path:
        # Group data by task_name
        grouped_data = {}
        for item in processed_data_with_metric:
            task_name = item['task_name']
            if task_name not in grouped_data:
                grouped_data[task_name] = []
            grouped_data[task_name].append(item)
        
        # Save each group to a separate file
        for task_name, data in grouped_data.items():
            task_output_path = output_path.parent / f"{output_path.stem}_{task_name}{output_path.suffix}"
            with jsonlines.open(task_output_path, 'w') as writer:
                writer.write_all(data)
            print(f"Saved {len(data)} samples for {task_name} to {task_output_path}")
    


if __name__ == "__main__":
    main()