import os
from modal import (App, 
                   Secret, 
                   gpu, 
                   method,
                   build,
                   enter, 
                   Image)


app = App(name="outlines-app-regex")

outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "huggingface",
    "outlines==0.0.39",
    "transformers==4.40.2",
    "datasets==2.18.0",
    "accelerate==0.27.2",
    "einops==0.7.0",
    "sentencepiece==0.2.0",
)

MODEL_DIR = "/vol/models"
supported_models = [
    'mistralai/Mistral-7B-Instruct-v0.3',
    'mistralai/Mistral-7B-Instruct-v0.2',
    'microsoft/Phi-3-medium-4k-instruct',
                    ]

@app.cls(image=outlines_image, secrets=[Secret.from_name("wills-huggingface-secret")], gpu=gpu.A100(size="80GB",count=2), timeout=500)
class Model:

    # we can have the model_name and allow it to be None
    # in the None will be used for prefetching
    # the name itself will be used to preload
    def __init__(self) -> None:
        self.model = None

    @build()  # add another step to the image build
    def preload_models(self):
        # there's a better way to do this... but for now this should work.
        for model_path in supported_models:
            self.load_model(model_path)

    def load_model(self, model_name):
        import outlines
        import torch
        assert model_name in supported_models, f"{model_name} is not supported!"
        self.model = outlines.models.transformers(
            model_name,
            model_kwargs={
                "token": os.environ["HF_TOKEN"],
                "cache_dir": MODEL_DIR,
                "device_map": "auto",
                "trust_remote_code": True,
                'torch_dtype': torch.bfloat16,
            },
        )
        return self.model

    @method()
    def generate(self, model_name, struct_regex: str, prompt: str, whitespace_pattern: str = None, sampler_name = "greedy"):

        import outlines
        from outlines.samplers import greedy, beam_search, multinomial
        import torch

        sampler_map = {
            "greedy": lambda: greedy(),
            "multi8": lambda: multinomial(top_k=8),
            "multi4": lambda: multinomial(top_k=4),
            "multi2": lambda: multinomial(top_k=2),
            "beam32": lambda: beam_search(beams=32),
            "beam16": lambda: beam_search(beams=16),
            "beam8": lambda: beam_search(beams=8),
            "beam4": lambda: beam_search(beams=4),
        }

        sampler = sampler_map[sampler_name]
        if self.model is None:
            self.load_model(model_name)
        if whitespace_pattern:
            generator = outlines.generate.regex(self.model, struct_regex, whitespace_pattern=whitespace_pattern,sampler=sampler())
        else:
            generator = outlines.generate.regex(self.model, struct_regex,sampler=sampler())
        if "beam" in sampler_name:
            result = generator(prompt)[0]
        else:
            result = generator(prompt)
        return result