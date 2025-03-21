import json
import re
from collections import defaultdict

def load_data(file_path):
    """
    Load JSONL data from a file
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Error parsing line: {line}")
    return data


def calculate_accuracy(data):
    """
    Calculate accuracy metrics for problems
    
    Args:
        data: List of problem data dictionaries
    
    Returns:
        overall_accuracy: The fraction of correct answers
        first_attempt_accuracy: The fraction of correct first attempts
        final_attempt_accuracy: The fraction of correct final attempts
        problem_stats: Dictionary with statistics per problem ID
    """
    correct_count = 0
    total_count = 0
    
    # First collect all unique questions while preserving order
    unique_questions = []
    seen_questions = set()  # Helper set for O(1) lookup
    for item in data:
        if item.get('id'):
            question_id = item.get('id')
        else:
            question_id = item.get('question')
        if question_id not in seen_questions:
            unique_questions.append(question_id)
            seen_questions.add(question_id)
    
    # Create mapping from question to numeric ID based on order of appearance
    question_to_id = {q: i for i, q in enumerate(unique_questions)}
    
    # Group problems by numeric ID and track attempts
    problems = defaultdict(list)
    first_attempts_correct = 0
    total_problems = 0
    seen_problems = set()
    last_attempts = {}
    
    for item in data:
        # breakpoint()
        if item.get('id'):
            original_id = item.get('id')
        else:
            original_id = item.get('question')
        numeric_id = question_to_id[original_id]
        problems[numeric_id].append(item)
        
        # Track first attempt accuracy
        if numeric_id not in seen_problems:
            seen_problems.add(numeric_id)
            total_problems += 1
            print(f"first attempt: id={numeric_id}, correct={item.get('correct')}")
            if item.get('correct'):
                first_attempts_correct += 1
        
        # Always update last attempt for this problem
        last_attempts[numeric_id] = item.get('correct')
    
    for _id, attempt in last_attempts.items():
        print(f"last attempt: id={_id}, correct={attempt}")
    # Calculate statistics for each problem
    problem_stats = {}
    for numeric_id, instances in problems.items():
        problem_correct = 0
        correct_labels = []
        
        for instance in instances:
            is_correct = instance.get('correct')
            correct_labels.append(int(is_correct))
            # breakpoint()
            if is_correct:
                problem_correct += 1
                correct_count += 1
            total_count += 1
        
        accuracy = problem_correct / len(instances) if instances else 0
        problem_stats[numeric_id] = {
            'total': len(instances),
            'correct': problem_correct,
            'accuracy': accuracy,
            'correct_labels': correct_labels
        }
    
    overall_accuracy = correct_count / total_count if total_count > 0 else 0
    first_attempt_accuracy = first_attempts_correct / total_problems if total_problems > 0 else 0
    
    # Calculate final attempt accuracy
    final_attempts_correct = sum(1 for is_correct in last_attempts.values() if is_correct)
    final_attempt_accuracy = final_attempts_correct / total_problems if total_problems > 0 else 0
    
    return overall_accuracy, first_attempt_accuracy, final_attempt_accuracy, problem_stats


def generate_report(overall_accuracy, first_attempt_accuracy, final_attempt_accuracy, problem_stats):
    """
    Generate a human-readable report of the accuracy analysis with visualization
    """
    report = []
    report.append("=" * 50)
    report.append("ACCURACY ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Overall Accuracy: {overall_accuracy:.2%}")
    report.append(f"First Attempt Accuracy: {first_attempt_accuracy:.2%}")
    report.append(f"Final Attempt Accuracy: {final_attempt_accuracy:.2%}")

    sorted_problems = sorted(
        problem_stats.items(), 
        key=lambda x: x[0], 
        reverse=True
    )
    
    for problem_id, stats in sorted_problems:
        report.append(f"  {problem_id}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    for problem_id, stats in sorted_problems:
        report.append(f"  {problem_id}: {stats['correct_labels']}")
    
    return "\n".join(report)

def main(file_path):
    """
    Main function to run the accuracy analysis
    """
    # Load data
    print(f"Loading data from {file_path}...")
    data = load_data(file_path)
    print(f"Loaded {len(data)} records")
    
    # breakpoint()

    # Calculate accuracy
    print("Calculating accuracy...")
    overall_accuracy, first_attempt_accuracy, final_attempt_accuracy, problem_stats = calculate_accuracy(data)
    
    # Generate report
    report = generate_report(overall_accuracy, first_attempt_accuracy, final_attempt_accuracy, problem_stats)
    print("\n" + report)
    
    # Optional: Save report to file
    with open("accuracy_report.txt", "w") as f:
        f.write(report)
    
    return overall_accuracy, first_attempt_accuracy, final_attempt_accuracy, problem_stats

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to your JSONL file: ")
    
    main(file_path)