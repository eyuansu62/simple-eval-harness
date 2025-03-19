from math_verify import parse, verify

### Helper Functions
def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

class DatasetEvaluator:
    """Base class defining common interface"""
    def __init__(self, name, cache_name=None):
        self.name = name
        self.cache_name = cache_name
        
    def evaluate(self, prediction, answer):
        """Evaluate single prediction"""
        result_boxed = self.process_prediction(prediction)

        if not result_boxed:
            # print(f"[debug] result_boxed is empty; prediction={prediction}")
            return 0
        
        extracted_result = parse(f"${result_boxed}$")
        extracted_answer = parse(f"${answer}$")

        # Use the math evaluation subprocess.
        if verify(extracted_result, extracted_answer):
            return 1
        else:
            return 0

    def process_prediction(self, prediction):
        """Process model output"""
        result_boxed = last_boxed_only_string(prediction)
        return result_boxed
        
    def get_name(self):
        """Get dataset name"""
        return self.name

