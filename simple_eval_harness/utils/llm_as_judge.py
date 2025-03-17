class OmniJudge():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_context(
        self,
        question,
        reference_answer,
        student_solution,
    ):
        context = [
            {
                "role": "system",
                "content": "You are an experienced teacher in the field of MATHEMATICS."
            },
            {
                "role": "user",
                "content": "# OBJECTIVE #\nYou are tasked with evaluating the correctness of a student's answer. Below, you are provided with a problem, a reference answer, and a student's answer. You should assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.\n\nYour tasks include:\nA. Identify Mathematical or Notational Equivalence.\nB. Conclude with a brief explanation as to why the student's output is correct or incorrect.\n\n# RESPONSE: MARKDOWN REPORT #\n## Student Final Answer\n[Extract the student's final answer, which is enclosed in \"\\\\boxed{}\".]\n## Equivalence Judgement\n[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]\n## Justification\n[Conclude with a brief explanation as to why the student's answer is correct or incorrect.]\n\n\n# ATTENTION #\n - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.\n - The answer is FALSE even if the student's final answer almost correct with a minor mistakes.\n - The answer is contained within the \"boxed\" section, so you can focus solely on comparing the content in the student's answer box with the reference answer, without needing to consider the intermediate steps.\n - Add \"=== report over ===\" at the end of the report."
            },
            {
                "role": "assistant",
                "content": "OK."
            },
            {
                "role": "user",
                "content": "**Question**:\n{}\n\n**Reference Answer**\n{}\n\n**Student Solution**:\n{}\n\n".format(
                    "The numbers $1447$ , $1005$ and $1231$ have something in common: each is a $4$ -digit number beginning with $1$ that has exactly two identical digits. How many such numbers are there?",
                    "432",
                    "To determine how many 4-digit numbers beginning with 1 have exactly two identical digits, we need to consider the structure of these numbers. A number of this form can be represented as \\(1abc\\), where \\(a\\), \\(b\\), and \\(c\\) are digits from 0 to 9, and exactly one of the digits \\(a\\), \\(b\\), or \\(c\\) is 1, while the other two are different from each other and different from 1.\n\nWe can break down the problem into the following steps:\n\n1. **Choose the digit that is 1**: There are 3 positions (a, b, c) where the digit 1 can be placed. So, there are 3 choices for the position of the digit 1.\n\n2. **Choose the other two digits**: The remaining two digits must be different from each other and different from 1. Therefore, we have 9 choices for the first of these two digits (0-9 except 1) and 8 choices for the second digit (0-9 except 1 and the first digit). This gives us \\(9 \\times 8 = 72\\) choices for the other two digits.\n\nCombining these choices, the total number of 4-digit numbers beginning with 1 that have exactly two identical digits is:\n\\[3 \\times 72 = 216.\\]\n\nThus, the number of such numbers is \\(\\boxed{216}\\).",
                )
            },
            {
                "role": "assistant",
                "content": "## Student Final Answer\n{}\n\n## Equivalence Judgement\n{}\n\n## Justification\n{}\n\n=== report over ===".format(
                    "216",
                    "FALSE",
                    "The student's answer of 216 is incorrect in the context of the problem, which asks for the total count of 4-digit numbers beginning with 1 that have exactly two identical digits. The reference answer is 432.\nIn the student's solution, they consider only cases where the digit '1' is one of the identical digits. However, the problem also includes the scenario where the identical digits could be different from '1'. Thus, the student's calculation does not account for all valid configurations. The discrepancy in figures indicates that the student's answer does not share the same meaning as the reference answer.",
                )
            },
            {
                "role": "user",
                "content": "**Question**:\n{}\n\n**Reference Answer**\n{}\n\n**Student Solution**:\n{}\n\n".format(
                    question.strip(),
                    reference_answer.strip(),
                    student_solution.strip(),
                )
            },
            {
                "role": "assistant",
                "content": "## Student Final Answer"
            }
        ]

        # formatted_context = self.apply_chat_template(
        #     context,
        #     tokenize=False,
        #     add_generation_prompt=True
        # ) + "## Student Final Answer"

        return context
    
    def parse_response(
        self,
        response
    ):
        prediction = {
            "answer": None,
            "judgement": None,
            "justification": None,
        }
        
        # parse the response
        if "## Student Final Answer" in response:
            first_start = response.index("## Student Final Answer") + len("## Student Final Answer")
        else:
            first_start = 0
        
        if "## Equivalence Judgement" in response:
            first_end = response.index("## Equivalence Judgement")
            prediction["answer"] = response[first_start:first_end].strip()

            second_start = response.index("## Equivalence Judgement") + len("## Equivalence Judgement")
            if "## Justification" in response:
                second_end = response.index("## Justification")
                judgement = response[second_start:second_end].strip()
                if judgement in ["TRUE", "FALSE"]:
                    prediction["judgement"] = judgement

                third_start = response.index("## Justification") + len("## Justification")
                third_end = len(response)
                justification = response[third_start:third_end].strip()

                if "=== report over ===" in justification:
                    justification = justification[:justification.index("=== report over ===")].strip()
                if "##" in justification:
                    justification = justification[:justification.index("##")].strip()  
                
                prediction["justification"] = justification
        
        return prediction