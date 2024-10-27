from .agents import VisualAgent, LanguageAgent
from .constants import GENDER_BIAS_LANGUAGE, GENDER_BIAS_VISUAL
from diffusers import StableDiffusionXLPipeline
import torch


def run_gender_bias_pipe(query: str):
    torch.cuda.empty_cache()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
    ).to("cuda")
    image = pipe(query, num_inference_steps=30).images[0]
    torch.cuda.empty_cache()
    gender_vl = VisualAgent(persona=GENDER_BIAS_VISUAL)
    language_vl = LanguageAgent(persona=GENDER_BIAS_LANGUAGE)

    image.save("init.png")

    answer = gender_vl.run_agent(image=image)
    
    final_query = str(answer + "\n" + f"Reference Query: {query}")

    print(final_query)

    final_res = language_vl.run_agent(query=final_query)

    print(final_res)

    image_res = pipe(final_res, num_inference_steps=30).images[0]

    image_res.save("final.png")

    print("Successfully Removed Bias")





    