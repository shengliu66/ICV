from tasks.base import BaseProbInference
from fastchat.model.model_adapter import get_conversation_template

class JailBreakProbInferenceForStyle(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        return "sp"

    def dataset_signature(self):
        return {
            "sample": ("jailbreak", None, "train"),
            "result": ("jailbreak", None, "test"),
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            query = (e["en_jailbreak_negative"], e["en_jailbreak_positive"])
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
        
        
        neg, pos = query 

        conv_template = get_conversation_template('vicuna_v1.5')

        conv_template.append_message(conv_template.roles[0], f"{neg}")
        conv_template.append_message(conv_template.roles[1], f"")

        prompt = conv_template.get_prompt()

        conv_template = get_conversation_template('vicuna_v1.5')
        conv_template.append_message(conv_template.roles[0], f"{neg}")
        conv_template.append_message(conv_template.roles[1], f"{pos}")
        
        prompt_both = conv_template.get_prompt()

        with_sentence = prompt
        with_sentence_and_paraphrase = prompt_both

        if return_reference:
            return with_sentence, with_sentence_and_paraphrase, pos
        else:
            return with_sentence, with_sentence_and_paraphrase