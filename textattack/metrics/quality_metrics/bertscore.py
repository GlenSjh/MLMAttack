"""

BertScoreMetric class:
-------------------------------------------------------
Class for calculating BertScore similarity on AttackResults

"""

import textattack
from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric
import bert_score


class BertScoreMetric(Metric):
    def __init__(self, **kwargs):
        self.bert_scorer = bert_score.BERTScorer(
            model_type='bert-base-uncased', idf=False, device=textattack.shared.utils.device, num_layers=12
        )
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results):
        """Calculates average USE similarity on all successfull attacks

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset

        Example::


            >> import textattack
            >> import transformers
            >> model = transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            >> tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            >> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
            >> attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
            >> dataset = textattack.datasets.HuggingFaceDataset("glue", "sst2", split="train")
            >> attack_args = textattack.AttackArgs(
                num_examples=1,
                log_to_csv="log.csv",
                checkpoint_interval=5,
                checkpoint_dir="checkpoints",
                disable_stdout=True
            )
            >> attacker = textattack.Attacker(attack, dataset, attack_args)
            >> results = attacker.attack_dataset()
            >> usem = textattack.metrics.quality_metrics.BertScoreMetric().calculate(results)
        """

        self.results = results

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                self.original_candidates.append(result.original_result.attacked_text.text.lower())
                self.successful_candidates.append(result.perturbed_result.attacked_text.text.lower())

        score = self.calc_bs(self.original_candidates, self.successful_candidates)

        self.all_metrics["avg_attack_bert_score"] = round(
            sum(score)/len(score), 2
        )

        return self.all_metrics

    def calc_bs(self, orig_texts, ref_texts):
        return self.bert_scorer.score(orig_texts, ref_texts)[-1].tolist()
