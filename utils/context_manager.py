import os
from contextlib import AbstractContextManager, ExitStack
from typing import Iterable
from utils.forward_tracer import ForwardTracer, ForwardTrace 


class CombinedContextManager(AbstractContextManager):
    def __init__(self, context_managers):
        self.context_managers = context_managers
        self.stack = None

    def __enter__(self):
        self.stack = ExitStack()
        for cm in self.context_managers:
            self.stack.enter_context(cm)
        return self.stack

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stack is not None:
            self.stack.__exit__(exc_type, exc_val, exc_tb)


def modified_forward_context_manager(model, forward_modifiers=()):
    context_manager = CombinedContextManager([*forward_modifiers])
    return context_manager

def traced_forward_context_manager(model, with_submodules=False):
    forward_trace = ForwardTrace()
    context_manager = ForwardTracer(model, forward_trace, with_submodules=with_submodules)
    return context_manager, forward_trace