from .evaluator import DatasetEvaluator
from simple_eval_harness.utils.llm_as_judge import OmniJudge
from simple_eval_harness.template_api import TemplateAPI

class OminiMathEvaluator(DatasetEvaluator):
    def __init__(self, name="ominimath"):
        super().__init__(name)

        self.judge_helper = OmniJudge()
        self.sampling_params = {
            "temperature": 0.9,
            "top_p": 0.9,
            "max_tokens": 8192,
        }
        self.judge_llm = TemplateAPI(
        )

    def evaluate_by_judge(self, predictions, answers, questions=None):
        prompts = []
        for prediction, answer, question in zip(predictions, answers, questions):
            prompts.append(self.judge_helper.get_context(question=question, reference_answer=answer, student_solution=prediction))
        
        responses = self.judge_llm.generate_until(prompts, gen_kwargs=self.sampling_params)
        responses = [self.judge_helper.parse_response(response) for response in responses]
        return responses
