import pandas as pd
from tqdm import tqdm

csv_path = "/home/forte/users/nishida/dataset/vqa/vqa_mscocotifa_train.csv"
output_path_5w1H = "/home/forte/users/nishida/dataset/vqa-sep/vqa-5w1h_mscoco.csv"
output_path_yes = "/home/forte/users/nishida/dataset/vqa-sep/vqa-yes_mscoco.csv"

def split_csv(path,output_path_5w1H,output_path_yes):

    vqa_5w1h = pd.DataFrame(columns = ["path","question","answer"])
    vqa_yes = pd.DataFrame(columns = ["path","question","answer"])
    source = pd.read_csv(path)

    for i in tqdm(range(len(source))):
        image_path = source["path"][i]
        question = source["question"][i]
        answer = source["answer"][i]

        if answer == "yes":
            vqa_yes = vqa_yes.append({"path":image_path,"question":question,"answer":answer},ignore_index=True)
        
        else:
            vqa_5w1h = vqa_5w1h.append({"path":image_path,"question":question,"answer":answer},ignore_index=True)

    #print(vqa_5w1h)
    #print(vqa_yes)
    vqa_yes.to_csv(output_path_yes,index=False)
    vqa_5w1h.to_csv(output_path_5w1H,index=False)


split_csv(csv_path,output_path_5w1H,output_path_yes)


csv_path = "/home/forte/users/nishida/dataset/vqa/vqa_flikcrtifa_train.csv"
output_path_5w1H = "/home/forte/users/nishida/dataset/vqa-sep/vqa-5w1h_flickr.csv"
output_path_yes = "/home/forte/users/nishida/dataset/vqa-sep/vqa-yes_flickr.csv"

split_csv(csv_path,output_path_5w1H,output_path_yes)