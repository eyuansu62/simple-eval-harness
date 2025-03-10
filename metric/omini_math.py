from .evaluator import DatasetEvaluator


class OminiMathEvaluator(DatasetEvaluator):
    def __init__(self, name="ominimath"):
        super().__init__(name)
        
