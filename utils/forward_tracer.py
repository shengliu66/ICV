from dataclasses import asdict, dataclass
from typing import Dict, Optional
import torch
from transformers import PreTrainedModel
from utils.llm_layers import get_embedding_layer, get_layers, get_attention_layers, get_mlp_layers


@dataclass
class ResidualStream:
    hidden: torch.Tensor
    attn: torch.Tensor
    mlp: torch.Tensor


class ForwardTrace:
    def __init__(self):
        self.residual_stream: Optional[ResidualStream] = ResidualStream(
            hidden=[],
            attn=[],
            mlp=[],
        )
        self.attentions: Optional[torch.Tensor] = None


class ForwardTracer:
    def __init__(self, model: PreTrainedModel, forward_trace: ForwardTrace, with_submodules: bool = False):
        self._model = model
        self._forward_trace = forward_trace
        self._with_submodules = with_submodules

        self._layers = get_layers(model)
        self._attn_layers = get_attention_layers(model)
        self._mlp_layers = get_mlp_layers(model)

        self._hooks = []

    def __enter__(self):
        self._register_forward_hooks()

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()

        if exc_type is None:
            residual_stream = self._forward_trace.residual_stream

            if residual_stream.hidden[0] == []:
                residual_stream.hidden.pop(0)

            for key in residual_stream.__dataclass_fields__.keys():
                acts = getattr(residual_stream, key)
                # TODO: this is a hack, fix it
                if key != "hidden" and not self._with_submodules:
                    continue

                nonempty_layer_acts = [layer_acts for layer_acts in acts if layer_acts != []][0]
                final_shape = torch.cat(nonempty_layer_acts, dim=0).shape

                for i, layer_acts in enumerate(acts):
                    if layer_acts == []:
                        acts[i] = torch.zeros(final_shape)
                    else:
                        acts[i] = torch.cat(layer_acts, dim=0)
                acts = torch.stack(acts).transpose(0, 1)
                setattr(residual_stream, key, acts)


            # if self._with_submodules:
            #     self._forward_trace.attentions = torch.stack(self._forward_trace.attentions).transpose(0, 1)
            # else:
            self._forward_trace.attentions = None

    def _register_forward_hooks(self):
        model = self._model
        hooks = self._hooks

        residual_stream = self._forward_trace.residual_stream

        def store_activations(residual_stream: ResidualStream, acts_type: str, layer_num: int):
            def hook(model, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                out = out.float().to("cpu", non_blocking=True)

                acts = getattr(residual_stream, acts_type)
                while len(acts) < layer_num + 1:
                    acts.append([])
                try:
                    acts[layer_num].append(out)
                except IndexError:
                    print(len(acts), layer_num)

            return hook

        def store_attentions(layer_num):
            def hook(model, inp, out):
                attention_maps = out[1]
                attention_maps = attention_maps.to("cpu", non_blocking=True).float()
                self._forward_trace.attentions[layer_num] = attention_maps

            return hook

        embedding_hook = get_embedding_layer(self._model).register_forward_hook(
            store_activations(residual_stream, "hidden", 0)
        )
        hooks.append(embedding_hook)

        for i, layer in enumerate(self._layers):

            hidden_states_hook = layer.register_forward_hook(store_activations(residual_stream, "hidden", i + 1))
            hooks.append(hidden_states_hook)

        if self._with_submodules:
            for i, mlp_layer in enumerate(self._mlp_layers):
                mlp_res_hook = mlp_layer.register_forward_hook(store_activations(residual_stream, "mlp", i))
                hooks.append(mlp_res_hook)

            for i, attn_layer in enumerate(self._attn_layers):
                attn_res_hook = attn_layer.register_forward_hook(store_activations(residual_stream, "attn", i))
                hooks.append(attn_res_hook)
                # attn_attentions_hook = attn_layer.register_forward_hook(store_attentions(i))
                # hooks.append(attn_attentions_hook)