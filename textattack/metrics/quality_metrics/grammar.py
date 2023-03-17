"""

Grammar class:
-------------------------------------------------------
Class for calculating Grammar error on AttackResults

"""

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric
import language_tool_python

class GrammarMetric(Metric):
    def __init__(self, **kwargs):
        self.lang_tool = language_tool_python.LanguageTool('en-US')
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
            >> usem = textattack.metrics.quality_metrics.GrammarMetric().calculate(results)
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

        grammar_orig = self.calc_grammar(self.original_candidates)
        grammar_attack = self.calc_grammar(self.successful_candidates)

        self.all_metrics["avg_original_grammar"] = round(sum(grammar_orig) / len(grammar_orig), 2)

        self.all_metrics["avg_attack_grammar"] = round(sum(grammar_attack) / len(grammar_attack), 2)

        return self.all_metrics

    def calc_grammar(self, texts):
        errors = [len(self.lang_tool.check(sent)) for sent in texts]
        return errors