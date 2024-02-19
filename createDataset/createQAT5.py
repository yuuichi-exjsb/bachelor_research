#形態素解析+T5による質疑応答生成

import spacy
from transformers import AutoModelWithLMHead, AutoTokenizer
import pandas as pd
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

vqa_all_df = pd.DataFrame(columns = ["path","question","answer"])

def get_question(answer, context, max_length=64):
  input_text = "answer: %s  context: %s </s>" % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])

# spacyモデルを読み込む
nlp = spacy.load("en_core_web_sm")
'''


#流れ
#dataframeから要素抽出
#captionに対して 固有表現抽出
#単語とquesionのリストができる
#ランダムにいっこ選ぶ

#回答と問題として、pathと一緒に格納


caption_df = pd.read_csv("/home/forte/users/nishida/cap-flickr.csv")

for i in tqdm(range(len(caption_df))):
   path = caption_df["path"][i]
   context = caption_df["caption"][i]
   #print("caption:",context)

   doc = nlp(context)
   for token in doc:
    if token.pos_ == "NOUN":
        #print("answer is " + token.text)
        #print("question is " + get_question(token.text,context))
        #print("===============")

        vqa_all_df = vqa_all_df.append({"path":path,"question":get_question(token.text,context),"answer":token.text},ignore_index=True)

vqa_all_df.to_csv("/home/forte/users/nishida/vqa-flickr.csv",index=False)
'''