from PIL import Image
import torch
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


class VisualAgent(object):
    def __init__(self, persona: str, temperature: int = 0) -> None:
        self.persona = persona
        self.temperature = temperature
    
    def run_agent(self, images: Image) -> str:
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf"
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, device_map="auto")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{self.persona}"},
                ]
            },
        ]
        conversation_1 = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{self.persona}"},
                ]
            },
        ]
        conversation_2 = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{self.persona}"},
                ]
            },
        ]
        
        prompt = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        prompt1 = processor.apply_chat_template(
            conversation_1, 
            add_generation_prompt=True
        )
        prompt2 = processor.apply_chat_template(
            conversation_2, 
            add_generation_prompt=True
        )
        
        processor.tokenizer.padding_side = "left"
        inputs = processor(
            images=images, 
            text=[prompt, prompt1, prompt2], 
            return_tensors="pt"
        ).to("cuda:0")
        
        output = model.generate(**inputs, max_new_tokens=300)
        result_ = processor.decode(output[0], skip_special_tokens=True)
        result__ = processor.decode(output[1], skip_special_tokens=True)
        result___ = processor.decode(output[2], skip_special_tokens=True)


        print(len(output))

        decoded_out = str(result_).split("[/INST]")[-1]
        decoded_out_1 = str(result__).split("[/INST]")[-1]
        decoded_out_2 = str(result___).split("[/INST]")[-1]
        
        final_answer = "Image 1:"+ decoded_out + "\n" + "Image 2:" + decoded_out_1 + "\n" + "Image 3:" + decoded_out_2

        print("+"*100)
        print(final_answer)
        print("+"*100)
        return final_answer



class LanguageAgent(object):
    def __init__(self, persona: str, temperature: int = 0) -> None:
        self.persona = persona
        self.temperature = temperature
    

    def run_agent(self, query: str) -> str:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )
        messages = [
            {
                "role": "system", 
                "content": f"{self.persona}"
            },
            {
                "role": "user",
                "content": f"{query}"
            }
        ]
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        outputs = model.generate(inputs, max_new_tokens=300)
        
        result_ = tokenizer.decode(outputs[0])

        result_dict = str(result_).split("<|assistant|>")[-1]

        result_dict_ = str(result_dict).split("<|end|>")[0]


        return result_dict_