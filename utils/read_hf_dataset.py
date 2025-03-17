# Import required libraries
from datasets import load_dataset, get_dataset_config_names
import json

# Function to save dataset to JSONL file with split and config information
def save_to_jsonl(datasets_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        total_records = 0
        # Iterate through each config
        for config in datasets_dict:
            dataset = datasets_dict[config]
            # Iterate through each split in the config
            for split_name in dataset.keys():
                print(f"Processing {config} - {split_name} split...")
                for item in dataset[split_name]:
                    # Create a new dictionary with config and split information
                    record = {
                        "config": config,
                        "split": split_name
                    }
                    # Add all original fields to the record
                    record.update(item)
                    # Convert to JSON string and write with a newline
                    # Ensure ASCII is not enforced to support Chinese characters
                    json_string = json.dumps(record, ensure_ascii=False)
                    f.write(json_string + '\n')
                    total_records += 1
        return total_records

def main():
    try:
        # Get all available config names automatically
        print("Retrieving available configurations from Hugging Face...")
        config_names = get_dataset_config_names("UGPhysics/ugphysics")
        print(f"Found {len(config_names)} configurations: {config_names}")
        
        # Load all configs of the dataset
        print("Loading all dataset configurations...")
        datasets_dict = {}
        for config in config_names:
            print(f"Loading {config}...")
            datasets_dict[config] = load_dataset("UGPhysics/ugphysics", config)
        
        # Define output filename
        output_file = "ugphysics_all_configs_splits.jsonl"
        
        # Save all configs and splits to single JSONL file
        print("Saving to JSONL file with UTF-8 encoding (Chinese support)...")
        total_records = save_to_jsonl(datasets_dict, output_file)
        
        # Print summary
        print(f"Saved all configs and splits to {output_file}")
        print(f"Total number of records: {total_records}")
        print(f"Number of configs processed: {len(config_names)}")
        print("Configs included:", config_names)
        
        # Test Chinese character support
        print("\nTesting Chinese character support...")
        test_data = {"config": "测试", "split": "训练", "text": "这是一个中文测试"}
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(test_data, ensure_ascii=False) + '\n')
        print("Added test record with Chinese characters: ", test_data)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Install required package if not already installed
    try:
        import datasets
    except ImportError:
        print("Installing required package: datasets")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
        
    main()