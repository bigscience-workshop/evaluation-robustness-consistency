from jinja2 import Template
from evaluation.tasks.auto_task import AutoTask
from evaluation.tasks.mrpc_negative.mrpc_negative import MRPCNegativeTask

TEMPLATE_STD = Template(
    """
Sentence 1: {{sent1}}
Sentence 2: {{sent2}}
Do these two sentences express the same meaning? Yes or No?
    """
)

TEMPLATE_SWP = Template(
    """
Sentence 1: {{sent2}}
Sentence 2: {{sent1}}
Do these two sentences express the same meaning? Yes or No?
    """
)

class MRPCSwapTask(MRPCNegativeTask, AutoTask):
    TEMPLATE_1 = TEMPLATE_STD
    TEMPLATE_2 = TEMPLATE_SWP   
    
    def is_consistent(self, answer_1, answer_2):
      # swapping prompt, so consistent if the answer is the same
      return answer_1 == answer_2
    
    @staticmethod
    def get_display_name() -> str:
        return "mrpc_swap"
