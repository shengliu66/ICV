from tasks.base import BaseProbInference


class ParaDetoxProbInferenceForStyle(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        return "sp"

    def dataset_signature(self):
        return {
            "sample": ("s-nlp/paradetox", None, "train"),
            "result": ("s-nlp/paradetox", None, "train"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            query = (e["en_toxic_comment"], e["en_neutral_comment"])
            data.append({"query": query})
        return data

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        if self.prompt_version.startswith("sp"):
            return ".  "
        else:
            raise ValueError(f"AGNews: Not supported prompt_version: {self.prompt_version}")
        
    def paralell_style_promptify(self, query, return_reference = False, Instruction = ''):
        toxic, neutral = query

        with_sentence_and_paraphrase = Instruction + f'Original: "{toxic}"; Paraphrased: "{neutral}"'
        with_sentence = Instruction + f'Original: "{toxic}"; Paraphrased: "'

        if return_reference:
            return with_sentence, with_sentence_and_paraphrase, neutral
        else:
            return with_sentence, with_sentence_and_paraphrase