from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import datetime
import torch.nn as nn
import torch
import numpy as np

dt_now = datetime.datetime.now()

#BLIP関係
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

train_df = pd.read_csv("/home/forte/users/nishida/dataset/vqa/vqa_flickr_train.csv")
train_df = train_df.sample(frac=1)
train_df = train_df[:65000]
test_df = pd.read_csv("/home/forte/users/nishida/dataset/vqa/vqa_flickr_test.csv")
test_df = test_df[:15000]
loss_df = pd.DataFrame(columns=['epoch','train_loss', 'valid_loss'])
result_df = pd.DataFrame(columns=['image_id',"question",'correct',"generate"])


batch_size = 4
lr = 5e-5
EPOCHS = 10


class Mydataset(Dataset):
    def __init__(self,processor):
        super().__init__()
        
        self.paths = train_df["path"].tolist()
        self.question = train_df["question"].tolist()
        self.answer = train_df["answer"].tolist()
        self.processor = processor
        
    def __len__(self):
        return len(self.paths)
    
    
    def __getitem__(self,idx):
        img_path = self.paths[idx]
        image = Image.open(img_path)
        question = self.question[idx]
        answer = self.answer[idx]
        answer = str(answer)
        
        inputs = self.processor(images=image
                                  , text=question, 
                                  padding="max_length", 
                                  return_tensors="pt")
        
        labels = self.processor(text = answer,
                                padding="max_length", 
                                return_tensors="pt").input_ids
        
        inputs["labels"] = labels
        
        inputs = {k:v.squeeze() for k,v in inputs.items()}

        return inputs
    

train_dataset = Mydataset(processor)
train_size = int(0.8*len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, valid_size]
)


train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model.to(device)

#PAD_INDEX = 0
#criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

best_valid_loss = float("inf")

for epoch in range(EPOCHS):
  print("EPOCHS:",epoch+1)
  model.train()
  epoch_train_loss = 0
  for batch in tqdm(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)
    attention_mask = batch.pop("attention_mask").to(device)
    labels = batch.pop('labels').to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask = attention_mask,
                    labels=labels)
    


    loss = outputs.loss
    epoch_train_loss += (loss.item())
    #print("Loss:", loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

  model.eval()
  epoch_valid_loss = 0

  with torch.no_grad():
    for idx, batch in enumerate(tqdm(valid_dataloader)):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        labels = batch.pop('labels').to(device)

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask = attention_mask,
                        labels=labels)
        
        loss = outputs.loss
        #print("Valid Loss:", loss.item())  

        epoch_valid_loss += loss.item()

    print("Train Loss:",epoch_train_loss/len(train_dataloader.dataset))
    print("Valid Loss:",epoch_valid_loss/len(valid_dataloader.dataset))
    loss_df = loss_df.append({'epoch':epoch+1,'train_loss': epoch_train_loss/len(train_dataloader.dataset), 'valid_loss': epoch_valid_loss/len(valid_dataloader.dataset)}, ignore_index=True)

    valid_ave = epoch_valid_loss/len(valid_dataloader.dataset)
    if best_valid_loss >= valid_ave:
       best_valid_loss = valid_ave
       torch.save(model.state_dict(), 'vqamscoco-tifa_weight.pth')

loss_df.to_csv("/home/forte/users/nishida/loss_vqa.csv",index=False)
loss_df.to_csv("/home/forte/users/nishida/loss-vqa/{}.csv".format(dt_now))





loss_csv = pd.read_csv("/home/forte/users/nishida/loss_vqa.csv")
x = loss_csv["epoch"].values
y1 = loss_csv["train_loss"].values
y2 = loss_csv["valid_loss"].values

# Set background color to white
fig = plt.figure()
fig.patch.set_facecolor('white')

# Plot lines
plt.xlabel('epoch')
plt.plot(x, y1, label='train_loss')
plt.plot(x, y2, label='valid_loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.xticks(np.arange(1, EPOCHS+1, step=1))
plt.legend()

# Visualize and save
plt.savefig("/home/forte/users/nishida/graph-vqa/{}.png".format(dt_now))
plt.show()

model.load_state_dict(torch.load('vqaflickr-_weight.pth'))

model.to(device)

with torch.no_grad():
   test_loss = 0
   for i in tqdm(range(len(test_df))):
      image_id = test_df["path"][i]
      question = test_df["question"][i]
      answer = test_df["answer"][i]

      image = Image.open(image_id)
      encoding = processor(images=image, text=question, padding="max_length",return_tensors="pt").to(device)

      outputs = model.generate(**encoding)
      generated  = processor.decode(outputs[0], skip_special_tokens=True)


      result_df = result_df.append({'image_id':image_id,'question': question,'correct':answer,'generate':generated},ignore_index = True)
    #print(generated)
result_df.to_csv("/home/forte/users/nishida/result/vqa_flickr-2024.csv",index=False)