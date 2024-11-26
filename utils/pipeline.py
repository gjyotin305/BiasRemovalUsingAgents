from .agents import VisualAgent, LanguageAgent
from .constants import LANGUAGE, GENDER_BIAS_VISUAL
from diffusers import StableDiffusionXLPipeline
import torch


def run_gender_bias_pipe(query: str):
    torch.cuda.empty_cache()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
    ).to("cuda")
    image = pipe(
        query,
        width=512,
        height=512, 
        num_inference_steps=20, 
        num_images_per_prompt=3
    ).images

    torch.cuda.empty_cache()
    
    gender_vl = VisualAgent(persona=GENDER_BIAS_VISUAL)
    language_vl = LanguageAgent(persona=LANGUAGE)

    for i, x in enumerate(image):
        x.save(f"init_{i}.png")

    answer = gender_vl.run_agent(images=image)
    
    final_query = str(answer + "\n" + f"Reference Query: {query}")

    print(final_query)

    final_res = language_vl.run_agent(query=final_query)

    print(final_res)

    image_res = pipe(final_res, num_inference_steps=30).images[0]

    image_res.save("final.png")

    print("Successfully Removed Bias")





    