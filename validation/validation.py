from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from tqdm import tqdm


from bert_score import score
import torchvision
from torchmetrics.multimodal.clip_score import CLIPScore
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")


device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("/home/forte/users/nishida/result/caption-kizon/flickr-ft.csv")
df.sample(frac=1)
df = df[:1000]

##3.BERTscore
def calc_BERTscore(correct:str,pred:str):
    _,__,BERTscore = score([correct],[pred],lang='en',verbose=True)
    return BERTscore.item()

##4.元画像とのCLIP score

def calc_CLIPscore(image:Image,correct:str,pred:str):

    image = torchvision.transforms.functional.to_tensor(image)
    CLIPscore_correct = metric(image,correct)
    CLIPscore_pred = metric(image,pred)

    return CLIPscore_correct.item(),CLIPscore_pred.item()



#path  = df["image_id"].tolist()
correct = df["correct"].tolist()
generate = df["generate"].tolist()

P, R, F1 = score(correct,generate, lang="en", verbose=True)
#print(P)
#print(R)
#print(F1)

print(torch.mean(input=P))
print(torch.mean(input=R))
print(torch.mean(input=F1))


"""
total_CLIPscore_correct = 0
total_CLIPscore_pred = 0
for i in tqdm(range(len(df))):
    #image = Image.open(path[i])
    correct_sentence = correct[i]
    generate_sentence = generate[i]

    _total_CLIPscore_correct,_total_CLIPscore_pred = calc_CLIPscore(image,correct_sentence,generate_sentence)

    total_CLIPscore_correct += _total_CLIPscore_correct
    total_CLIPscore_pred += _total_CLIPscore_pred

#print("CLIPscore_correct is :",total_CLIPscore_correct/len(df))
#print("CLIPscore_pred is ;",total_CLIPscore_pred/len(df))
"""



