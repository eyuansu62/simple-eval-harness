import argparse
import json
import jsonlines
from pathlib import Path
import threading
import requests
import time
import socket
import subprocess
import re
from template_api import TemplateAPI, CachingLM

# Simple configuration
PORT = 8000
TIMEOUT = 300
MODEL_BASE_PATH = "/share/project/huggingface/models"
CONFIG_PATH = "/home/qinbowen/subject_eval/API_tokens.json"
DEFAULT_INPUT = "/home/qinbowen/subject_eval/data/mt-input.jsonl"
OUTPUT_DIR = Path("clcc_mt")


def is_server_running(port=PORT):
    """Check if server is already running on the port"""
    # First check if port is in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('0.0.0.0', port)) != 0:
            return False  # Port is not in use
    
    # Port is in use, check if it's a responsive vLLM server
    try:
        response = requests.get(f"http://0.0.0.0:{port}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False


def start_server(model_path, port=PORT):
    """Start a vLLM server for the model"""
    # Determine tensor parallel size based on model name
    tp_size = 1
    if "B" in model_path:
        match = re.search(r'(\d+)B', model_path)
        if match and int(match.group(1)) > 27:
            tp_size = 8
    
    # Start the server process
    command = [
        "vllm", "serve", model_path,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", "0.9",
        "--trust-remote-code",
    ]
    print(f"Starting server: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    # Set up logging threads
    stop_event = threading.Event()
    for pipe in (process.stdout, process.stderr):
        if pipe:
            thread = threading.Thread(
                target=lambda p: [print(line, end="") for line in iter(p.readline, "") if not stop_event.is_set()],
                args=(pipe,)
            )
            thread.daemon = True
            thread.start()
    
    # Wait for server to become responsive
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        if is_server_running(port):
            print("Server is ready!")
            return process, stop_event
        time.sleep(2)
    
    # Server failed to start
    print(f"Server failed to start within {TIMEOUT} seconds")
    stop_event.set()
    process.terminate()
    return None, None


def stop_server(process, stop_event):
    """Stop the server"""
    if process is None:
        return
    
    stop_event.set()
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    print("Server stopped")


def load_config():
    """Load API configuration from file"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except:
        print(f"Could not load config from {CONFIG_PATH}")
        return {}

def process_predictions(prediction):
    """Process model output, handling DeepSeek format."""
    if prediction is None:
        return None
        
    try:
        match = re.search(r"</think>\n\n(.*)", prediction, re.DOTALL)
        return match.group(1) if match else prediction
    except Exception:
        return prediction


def process_conversations(data, model_name, model_config, concurrency=2, use_cache=None):
    """Process all conversations with the model"""
    # Initialize model
    model = TemplateAPI(
        model=model_name,
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        num_concurrent=concurrency,
    )
    
    # Set up caching if requested
    if use_cache:
        print(f"Using cache: {use_cache}")
        model = CachingLM(model, use_cache)
    
    # Find maximum number of turns
    max_turns = 0
    for sample in data:
        turns = sample.get('turns', [])
        if isinstance(turns, list):
            max_turns = max(max_turns, len(turns))
    print(f"Max turns: {max_turns}")
    
    # Initialize conversation states
    messages_list = [[] for _ in data]
    
    # Process each turn
    for turn_idx in range(max_turns):
        # Collect samples for this turn
        batch_indices = []
        batch_prompts = []
        
        for i, sample in enumerate(data):
            turns = sample.get('turns', [])
            if turn_idx < len(turns):
                # Add user message
                messages_list[i].append({"role": "user", "content": turns[turn_idx]})
                batch_indices.append(i)
                batch_prompts.append((f"sample_{i}_turn_{turn_idx}", messages_list[i][:]))
        
        # Generate responses
        if batch_prompts:
            print(f"Turn {turn_idx}: Processing {len(batch_prompts)} conversations")
            batch_predictions = model.generate_until(
                batch_prompts, 
                gen_kwargs={"max_gen_toks": 20000}
            )
            
            # Add assistant responses
            for idx, prediction in zip(batch_indices, batch_predictions):
                
                messages_list[idx].append({"role": "assistant", "content": process_predictions(prediction)})
    
    # Prepare results
    results = []
    for i, sample in enumerate(data):
        # Extract assistant responses
        predictions = [msg["content"] for msg in messages_list[i] if msg["role"] == "assistant"]
        results.append({**sample, 'prediction': predictions})
    
    return results


def save_results(results, model_name):
    """Save results to file"""
    # Create safe filename
    safe_name = model_name.replace("/", "_")
    output_file = OUTPUT_DIR / f"{safe_name}_clcc_mt.jsonl"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Write results
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(results)
    
    print(f"Results saved to {output_file}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate models on conversations")
    parser.add_argument("--file_path", type=str, default=DEFAULT_INPUT,
                       help="Input JSONL file path")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name")
    parser.add_argument("--multiprocess_num", type=int, default=2,
                       help="Number of concurrent processes")
    args = parser.parse_args()
    
    # Get model configuration
    all_configs = load_config()
    model_config = all_configs.get(args.model_name)
    
    process = None
    stop_event = None
    
    try:
        # Check if we need to start a server
        if not model_config:
            # First check if a server is already running
            if is_server_running():
                print("Using existing server")
                model_config = {
                    "api_key": "vllm",
                    "base_url": f"http://0.0.0.0:{PORT}/v1/chat/completions"
                }
            else:
                # Start a new server
                model_path = f"{MODEL_BASE_PATH}/{args.model_name}"
                print(f"Starting server for {model_path}")
                process, stop_event = start_server(model_path)
                if not process:
                    print("Failed to start server")
                    return
                
                model_config = {
                    "api_key": "vllm",
                    "base_url": f"http://0.0.0.0:{PORT}/v1/chat/completions"
                }
        
        # Load data
        with jsonlines.open(args.file_path, 'r') as reader:
            data = list(reader)
        print(f"Loaded {len(data)} conversations")
        
        # Create cache name
        cache_name = f"clcc_mt_cache_{args.model_name.replace('/', '_')}"
        
        # Process data
        results = process_conversations(
            data, 
            args.model_name, 
            model_config, 
            concurrency=args.multiprocess_num,
            use_cache=cache_name
        )
        
        # Save results
        save_results(results, args.model_name)
        
    finally:
        # Clean up
        if process and stop_event:
            stop_server(process, stop_event)


if __name__ == "__main__":
    main()