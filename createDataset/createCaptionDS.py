#画像生成コード
#必要1



from datasets import load_dataset
import random
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import pandas as pd
from PIL import Image
from tqdm import tqdm
from IPython.display import clear_output

dataset = load_dataset("carlosejimenez/flickr30k_captions_simCSE", split="train")
#print(dataset["text"])



select = dataset["text"]

select_seed = 27000 #取り出したいサンプル数
select  = random.sample(dataset["text"], select_seed)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-2", revision="fp16", torch_dtype=torch.float16, use_auth_token="YOUR_TOKEN")
pipe.to("cuda")
if pipe.safety_checker is not None:
    pipe.safety_checker = lambda images, **kwargs: (images, False)

caption_df = pd.DataFrame(columns=['path', 'caption'])
generate_num = 3
fig_size = 384
for i,prompt in enumerate(tqdm(select)):
    for j in range(generate_num):
        with autocast("cuda"):
            image = pipe(prompt)[0][0]
            image = image.resize((fig_size,fig_size))
            #filename = "/home/forte/users/nishida/outputs/{}_{}.png".format(prompt,i+1)
            filename = "/home/forte/users/nishida/outputs-flickr/{}-{}.png".format(i+1,j+1)
            caption_df = caption_df.append({'path': filename, 'caption': prompt}, ignore_index=True)
        
        #image.save(f"/home/forte/users/nishida/outputs/{prompt}_{i+1}.png")
        image.save(f"/home/forte/users/nishida/outputs-flickr/{i+1}-{j+1}.png")
    clear_output()
        
caption_df.to_csv("/home/forte/users/nishida/cap-flickr.csv",index=False)