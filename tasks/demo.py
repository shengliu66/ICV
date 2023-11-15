from tasks.base import BaseProbInference


class DemoProbInferenceForStyle(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        return "sp"

    def dataset_signature(self):
        return {
            "sample": ("demo", None, "train"),
            "result": ("demo", None, "test"),
        }

    def dataset_preprocess(self, raw_data):
        pass

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        if self.prompt_version.startswith("sp"):
            return ".  "
        else:
            raise ValueError(f"AGNews: Not supported prompt_version: {self.prompt_version}")
        
    def paralell_style_promptify(self, query, return_reference = False, Instruction = ''):
        
        pass


            