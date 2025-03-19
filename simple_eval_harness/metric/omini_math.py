from .evaluator import DatasetEvaluator
from simple_eval_harness.utils.llm_as_judge import OmniJudge
from simple_eval_harness.template_api import TemplateAPI, CachingLM
from simple_eval_harness.utils.token_soup import get_token

class OminiMathEvaluator(DatasetEvaluator):
    def __init__(self, name="ominimath", cache_name=None):
        super().__init__(name)

        self.judge_helper = OmniJudge()
        self.sampling_params = {
            "temperature": 0.9,
            "top_p": 0.9,
            "max_tokens": 8192,
        }
        model_config, model_name = get_token(
            "gpt-4o-2024-05-13", 
            use_fuzzy=True, 
            threshold=80
        )
        self.judge_llm = TemplateAPI(
            model=model_name,
            api_key=model_config["api_key"],
            base_url=model_config["base_url"],
            num_concurrent=1,
        )
        self.judge_llm = CachingLM(self.judge_llm, cache_name)
        
    def evaluate_by_judge(self, predictions, answers, questions=None):
        prompts = []
        for prediction, answer, question in zip(predictions, answers, questions):
            prompts.append(self.judge_helper.get_context(question=question, reference_answer=answer, student_solution=prediction))
        
        responses = self.judge_llm.generate_until(prompts, gen_kwargs=self.sampling_params)
        responses = [self.judge_helper.parse_response(response)['judgement']=="TRUE" for response in responses]

        return responses

    def analyze_results(self, samples):
        # Track overall statistics
        overall_correct = sum(1 for sample in samples if sample['correct'])
        overall_total = len(samples)
        
        # Group samples by domain
        domain_results = {}
        for sample in samples:
            domains = sample.get('domain', ['unknown'])
            # Ensure domains is a list
            if isinstance(domains, str):
                domains = [domains]
            
            # Update stats for each domain this sample belongs to
            for domain in domains:
                if domain not in domain_results:
                    domain_results[domain] = {'correct': 0, 'total': 0}
                domain_results[domain]['total'] += 1
                if sample['correct']:
                    domain_results[domain]['correct'] += 1
        
        # Calculate statistics for each domain
        task_stats = {
            'accuracy': overall_correct / overall_total,
            'correct': overall_correct,
            'total': overall_total,
            **{
                domain: {
                    'accuracy': results['correct'] / results['total'],
                    'correct': results['correct'],
                    'total': results['total']
                } for domain, results in domain_results.items()
            }
        }
        
        return task_stats