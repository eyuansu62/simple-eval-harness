import json
import sys

# Get file paths from command-line arguments
file1 = "/home/qinbowen/simple-eval-harness/eval_output/processed_Qwen2.5-32B-Instruct_tem_0.6_repetition_1.0_True_aime25.jsonl"
file2 = "/home/qinbowen/simple-eval-harness/eval_output/processed_QwQ-32B_30000_True_aime25.jsonl"

# Initialize counter for differences
diff_count = 0

correct_count_1 = 0
correct_count_2 = 0
try:
    # Open both JSONL files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        # Read lines from both files simultaneously
        for line_num, (line1, line2) in enumerate(zip(f1, f2)):
            try:
                # Parse JSON objects from each line
                data1 = json.loads(line1)
                data2 = json.loads(line2)
                
                # Extract the "correct" key values (returns None if key is missing)
                correct1 = data1.get("correct")
                correct2 = data2.get("correct")
                # breakpoint()
                # Compare the "correct" values

                if correct1 != correct2:
                    breakpoint()
                    print(f"Line {line_num}: file1 has 'correct': {correct1}, file2 has 'correct': {correct2}")
                    diff_count += 1
                
                if correct1: correct_count_1 += 1
                if correct2: correct_count_2 += 1
            except json.JSONDecodeError:
                print(f"Line {line_num}: invalid JSON")
        

    # Print the total number of differences
    print(f"Total differences: {diff_count}")
    print(f"Correct count in file1: {correct_count_1}")
    print(f"Correct count in file2: {correct_count_2}")
except FileNotFoundError:
    print("One of the files does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")