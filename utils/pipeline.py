from .agents import VisualAgent, LanguageAgent
from .constants import LANGUAGE, BIAS, BIAS_QUERY, BIAS_CHECK_IMAGE, DEBIAS_QUERY_GEN
from diffusers import StableDiffusionXLPipeline
import torch


def run_gender_bias_pipe(query: str):
    torch.cuda.empty_cache()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
    ).to("cuda")
    
    image = pipe(
        query, 
        num_inference_steps=40, 
        num_images_per_prompt=3
    ).images

    torch.cuda.empty_cache()

    gender_vl = VisualAgent(persona=BIAS.format(query))
    language_vl = LanguageAgent(persona=BIAS_QUERY.format(query))
    language_vl_check = LanguageAgent(persona=BIAS_CHECK_IMAGE.format(query))
    language_vl_query = LanguageAgent(persona=DEBIAS_QUERY_GEN.format(query))

    for i, x in enumerate(image):
        x.save(f"init_{i}.png")

    print(len(image))

    check_answer = language_vl.run_agent(query="")

    print(check_answer)

    answer = gender_vl.run_agent(images=image)
    
    print(answer)

    final_answer = language_vl_check.run_agent(query=answer)

    print(final_answer)

    print(str(final_answer).split(" "))

    if str(final_answer).split(" ")[-1] == "Yes" or str(final_answer).split(" ")[-1] == "yes":
        new_query = language_vl_query.run_agent(query="")
        print(new_query)
        # cleaned_text = "".join(new_query.split())
        # print(new_query.split("|"))
        print(new_query.split("|"))
    else:
        new_query = query
    
    # final_res = language_vl.run_agent(query=final_query)

    # print(final_res)

    image_res = pipe(
        query, 
        num_inference_steps=30, 
        num_images_per_prompt=3
    ).images

    
    for i, x in enumerate(image_res):
        x.save(f"final_{i}.png")    

    print("Successfully Removed Bias")





    