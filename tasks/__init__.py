from .paradetox import ParaDetoxProbInferenceForStyle
from .shakespeare import ShakespeareProbInferenceForStyle
from .formality import FormalityProbInferenceForStyle
from .sentiment import SentimentProbInferenceForStyle
from .format import FormatProbInferenceForStyle
from .emotive import EmotiveProbInferenceForStyle
from .jailbreak import JailBreakProbInferenceForStyle
from .demo import DemoProbInferenceForStyle

task_mapper = {
    "paradetox": ParaDetoxProbInferenceForStyle,
    "shakespeare": ShakespeareProbInferenceForStyle,
    "formality": FormalityProbInferenceForStyle,
    "sentiment": SentimentProbInferenceForStyle,
    "format": FormatProbInferenceForStyle,
    "emotive": EmotiveProbInferenceForStyle,
    "jailbreak": JailBreakProbInferenceForStyle,
    'demo': DemoProbInferenceForStyle,
}


def load_task(name):
    if name not in task_mapper.keys():
        raise ValueError(f"Unrecognized dataset `{name}`")

    return task_mapper[name]
