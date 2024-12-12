'''"""
WMDP: Measuring and Reducing Malicious Use With Unlearning
https://arxiv.org/pdf/2403.03218.pdf

The Weapons of Mass Destruction Proxy (WMDP) benchmark is a dataset of 3,668 
multiple-choice  questions surrounding hazardous knowledge in biosecurity, cybersecurity, 
and chemical  security. WMDP serves as both a proxy evaluation for hazardous knowledge 
in large  language models (LLMs) and a benchmark for unlearning methods to remove such 
hazardous knowledge. To guide progress on mitigating risk from LLMs, we develop RMU, a 
state-of-the-art unlearning method which reduces model performance on WMDP while 
maintaining general language model capabilities.

Homepage: https://www.wmdp.ai/ 

_CITATION = """
@misc{li2024wmdp,
      title={The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning}, 
      author={Nathaniel Li and Alexander Pan and Anjali Gopal and Summer Yue and Daniel Berrios and Alice Gatti and Justin D. Li and Ann-Kathrin Dombrowski and Shashwat Goel and Long Phan and Gabriel Mukobi and Nathan Helm-Burger and Rassin Lababidi and Lennart Justen and Andrew B. Liu and Michael Chen and Isabelle Barrass and Oliver Zhang and Xiaoyuan Zhu and Rishub Tamirisa and Bhrugu Bharathi and Adam Khoja and Zhenqi Zhao and Ariel Herbert-Voss and Cort B. Breuer and Samuel Marks and Oam Patel and Andy Zou and Mantas Mazeika and Zifan Wang and Palash Oswal and Weiran Liu and Adam A. Hunt and Justin Tienken-Harder and Kevin Y. Shih and Kemper Talley and John Guan and Russell Kaplan and Ian Steneker and David Campbell and Brad Jokubaitis and Alex Levinson and Jean Wang and William Qian and Kallol Krishna Karmakar and Steven Basart and Stephen Fitz and Mindy Levine and Ponnurangam Kumaraguru and Uday Tupakula and Vijay Varadharajan and Yan Shoshitaishvili and Jimmy Ba and Kevin M. Esvelt and Alexandr Wang and Dan Hendrycks},
      year={2024},
      eprint={2403.03218},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
'''

from typing import Optional

from catwalk.dependencies.lm_eval.base import MultipleChoiceTask
from catwalk.task import rc_metrics
import numpy as np

'''class WMDPMCStd(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "cais/wmdp"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_metrics(rc_metrics(primary="acc_raw"))
    
    def _process_doc(self, doc):
        return {
            "query": doc["question"],
            "choices": doc["choices"]
        }

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["answer"]]

    def construct_requests(self, doc, ctx):
        return [
            rf.loglikelihood(ctx, " {}".format(choice))[0]
            for choice in doc["choices"]
        ]

    def process_results(self, doc, results):
        gold = doc["answer"]
        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self):
        return {
            "acc": np.mean,
            "acc_norm": np.mean,
        }
    
def create_wmdp_tasks():
    return {
    "wmdp_mc_std": WMDPMCStd().add_metrics(rc_metrics(primary="acc_per_char"))
    }
'''
'''
class WMDPStd(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "cais/wmdp"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return {
            "query": doc["question"],
            "choices": doc["choices"],
            "gold": doc["answer"],
        }

    #def fewshot_examples(self, k, rnd):
    #    if self._fewshot_docs is None:
    #    # Use 'test' split instead of 'train' for few-shot examples
    #        self._fewshot_docs: Optional[list] = list(map(self._process_doc, self.dataset["test"]))
    #    return rnd.sample(self._fewshot_docs, k)

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

class WMDPMCStd(WMDPStd):
    DATASET_NAME = "wmdp-bio"

    def _process_doc(self, doc):
        choices = doc["choices"]
        query = f"{doc['question']}\n"
        for i, choice in enumerate(choices):
            query += f" {chr(65+i)}. {choice}\n"
        query += "Answer:"
        
        return {
            "query": query,
            "choices": [chr(65+i) for i in range(len(choices))],
            "gold": doc["answer"],
        }
    
def create_wmdp_tasks():
    return {
        "wmdp_mc_std": WMDPMCStd().add_metrics(rc_metrics(primary="acc_per_char"))
    }

class WMDPMCStd(WMDPStd):
    def _process_doc(self, doc):
        choices = doc["choices"]
        query = f"{doc['question']}\n"
        for i, choice in enumerate(choices):
            query += f" {chr(65+i)}. {choice}\n"
        query += "Answer:"
        
        return {
            "query": query,
            "choices": [chr(65+i) for i in range(len(choices))],
            "gold": doc["answer"],
        }

def create_wmdp_tasks():
    return {
        "wmdp_mc_std": WMDPMCStd().add_metrics(rc_metrics(primary="acc_per_char"))
    }'''


class WMDPMCBioStd(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "cais/wmdp"
    DATASET_NAME = "wmdp-bio"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        choices = doc["choices"]
        query = f"{doc['question']}\n"
        for i, choice in enumerate(choices):
            query += f" {chr(65+i)}. {choice}\n"
        query += "Answer:"
        
        return {
            "query": query,
            "choices": [chr(65+i) for i in range(len(choices))],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

class WMDPMCChemStd(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "cais/wmdp"
    DATASET_NAME = "wmdp-chem"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        choices = doc["choices"]
        query = f"{doc['question']}\n"
        for i, choice in enumerate(choices):
            query += f" {chr(65+i)}. {choice}\n"
        query += "Answer:"
        
        return {
            "query": query,
            "choices": [chr(65+i) for i in range(len(choices))],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

class WMDPMCCyberStd(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "cais/wmdp"
    DATASET_NAME = "wmdp-cyber"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        choices = doc["choices"]
        query = f"{doc['question']}\n"
        for i, choice in enumerate(choices):
            query += f" {chr(65+i)}. {choice}\n"
        query += "Answer:"
        
        return {
            "query": query,
            "choices": [chr(65+i) for i in range(len(choices))],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

def create_wmdp_tasks():
    return {
        "wmdp_mc_bio_std": WMDPMCBioStd().add_metrics(rc_metrics(primary="acc_raw")),
        "wmdp_mc_chem_std": WMDPMCChemStd().add_metrics(rc_metrics(primary="acc_raw")),
        "wmdp_mc_cyber_std": WMDPMCCyberStd().add_metrics(rc_metrics(primary="acc_raw"))
    }