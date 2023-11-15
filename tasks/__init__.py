from .demo import DemoProbInferenceForStyle
task_mapper = {
    'demo': DemoProbInferenceForStyle,
}


def load_task(name):
    if name not in task_mapper.keys():
        raise ValueError(f"Unrecognized dataset `{name}`")

    return task_mapper[name]
