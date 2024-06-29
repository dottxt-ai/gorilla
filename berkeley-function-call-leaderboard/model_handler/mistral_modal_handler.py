# Structured Generation of Function Calls using Outlines ( https://github.com/outlines-dev/outlines )
# PoC Code: Connesson, Rémi. (Apr 2024). Outlines Function Call Gorilla Leaderboard Experiment. GitHub.
# https://github.com/remiconnesson/outlines-func-call-gorilla-leaderboard-experiment/tree/main.

from modal import Cls
import os, json
from textwrap import dedent
import re

from model_handler.constant import GORILLA_TO_OPENAPI
from model_handler.model_style import ModelStyle
from model_handler.utils import (
    _cast_to_openai_type,
    ast_parse,
)
from huggingface_hub import snapshot_download
from pathlib import Path


from mistral_common.protocol.instruct.tool_calls import Function, Tool

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


class MistralModalHandler:
    model_name: str
    model_style: ModelStyle

    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        model_map = {
            'mistral7bV3': 'mistralai/Mistral-7B-Instruct-v0.3',
        }
        self.model_name = model_name
        self.model_name_internal = model_map[model_name]
        self.model_style = ModelStyle.Outlines
        mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3') # update this we want to support more models
        mistral_models_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=self.model_name_internal,
                  allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
                  local_dir=mistral_models_path)
    
        self.tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")

    def format_result(self, result):
        # formats just in case.
        from ast import literal_eval
        import json
        cleaned = result.replace("_dot_",".")
        processed = cleaned.split(']')[0] + ']'
        try:
            # this should actually work most of the time
            value = json.loads(processed)[0]
        except:
            try:
                value = literal_eval(processed)[0]
            except:
                pass
        function_name = value['name']
        arg_values = value['arguments']
        args_string = ', '.join([f"{key}=\"{value}\"" if isinstance(value, str) else f"{key}={value}" for key, value in arg_values.items()])
        # Creating the output string with the function name and arguments
        output_string = f'[{function_name}({args_string})]'
        return output_string
    
    def inference(self, prompt, functions, test_category):
        # IMPORTANT: Only works for 'simple' test_category.
        # Assumes only 1 function to be called

        if type(functions) is not list:
                functions = [functions]
        function = functions[0]
        completion_request = ChatCompletionRequest(
            tools=[
                Tool(
                    function=Function(
                        name=re.sub("\.","_dot_",function['name']),
                        description=function['description'],
                        parameters={
                            "type": "object",
                            "properties": _cast_to_openai_type(function["parameters"]["properties"], GORILLA_TO_OPENAPI, "simple"),
                            "required": function['parameters']['required'],
                        },
                    )
                )
            ],
            messages=[
                UserMessage(content=prompt),
                ],
        )

        fc_prompt = self.tokenizer.encode_chat_completion(completion_request).text
        preformatted_result = None
        try:
            Model = Cls.lookup("outlines-app-text", "Model",environment_name="eval")
            m = Model()
            result = m.generate.remote(self.model_name_internal, fc_prompt)
            preformatted_result = result
            result = self.format_result(result)
        except Exception as e:
            result = f'[error.message(error="{e}", result={preformatted_result})]'
        return result, {"input_tokens": 0, "output_tokens": 0, "latency": 0}


    def decode_ast(self, result, language="Python"):
        decoded_output = ast_parse(result,language)
        return decoded_output

    def decode_execute(self, result):
        # This method takes raw model output and convert it to standard execute checker input.
        pass

    def write(self, result, file_to_open):
        # This method is used to write the result to the file.
        if not os.path.exists("./result"):
            os.mkdir("./result")
        if not os.path.exists("./result/" + self.model_name.replace("/", "_")):
            os.mkdir("./result/" + self.model_name.replace("/", "_"))
        with open(
            "./result/"
            + self.model_name.replace("/", "_")
            + "/"
            + file_to_open.replace(".json", "_result.json"),
            "a+",
        ) as f:
            f.write(json.dumps(result) + "\n")

    def load_result(self, test_category):
        # This method is used to load the result from the file.
        result_list = []
        with open(
            f"./result/{self.model_name.replace('/', '_')}/gorilla_openfunctions_v1_test_{test_category}_result.json"
        ) as f:
            for line in f:
                result_list.append(json.loads(line))
        return result_list