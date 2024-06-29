import os
from modal import (App, 
                   Secret, 
                   gpu, 
                   method,
                   build,
                   enter, 
                   Image)


app = App(name="outlines-app-text")

outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "huggingface",
    "outlines==0.0.39",
    "transformers==4.40.2",
    "datasets==2.18.0",
    "accelerate==0.27.2",
    "einops==0.7.0",
    "sentencepiece",
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
    def generate(self, model_name, prompt: str):

        import outlines
        from outlines.samplers import greedy, beam_search, multinomial
        import torch

        if self.model is None:
            self.load_model(model_name)

        generator = outlines.generate.text(self.model)
        result = generator(prompt)
        return result