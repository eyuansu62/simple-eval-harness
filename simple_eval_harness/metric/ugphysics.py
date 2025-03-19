from .evaluator import DatasetEvaluator


class UgPhysicsEvaluator(DatasetEvaluator):
    def __init__(self, name="ugphysics", cache_name=None):
        super().__init__(name, cache_name)
        
