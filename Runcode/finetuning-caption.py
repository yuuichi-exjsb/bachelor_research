# 30 45 171 200

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import datetime
import torch

dt_now = datetime.datetime.now()

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


train_path = "/home/forte/users/nishida/dataset/caption/cap_mscoco_train.csv"
test_path_mscoco = "/home/forte/users/nishida/dataset/caption/cap_mscoco_test.csv"
test_path_flickr = "/home/forte/users/nishida/dataset/caption/cap_flickr_test.csv"
weight_path = "/home/forte/users/nishida/weight/caption/mscoco.pth"
result_path_mscoco = "/home/forte/users/nishida/result/caption/mscoco-mscoco.csv"
result_path_flickr = "/home/forte/users/nishida/result/caption/mscoco-flickr.csv"


train_df = pd.read_csv(train_path)
train_df = (train_df.sample(frac=1))[:65000]
test_df_mscoco = pd.read_csv(test_path_mscoco)
test_df_flickr = pd.read_csv(test_path_flickr)
loss_df = pd.DataFrame(columns=['epoch','train_loss', 'valid_loss'])
result_df = pd.DataFrame(columns=['image_id','correct',"generate"])

for param in model.parameters():
    param.requires_grad = False
last_layer = list(model.children())[-1]
for param in last_layer.parameters():
    param.requires_grad = True

#len(train_df)

#train settings
EPOCHS = 20
batch_size = 4
lr = 5e-5
total = len(train_df)

class Mydataset(Dataset):
    def __init__(self,processor):
        super().__init__()
        self.paths = train_df["path"].tolist()
        self.captions = train_df["caption"].tolist()
        self.processor = processor
        
    def __len__(self):
        return len(self.paths)
    
    
    def __getitem__(self,idx):
        img_path = self.paths[idx]
        image = Image.open(img_path)
        caption = self.captions[idx]
        encoding = self.processor(images=image
                                  , text=caption, 
                                  padding="max_length", 
                                  return_tensors="pt")
        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        
        return encoding
    
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

best_valid_loss = float("inf")

for epoch in range(EPOCHS):
  print("EPOCHS:",epoch+1)
  model.train()
  epoch_train_loss = 0
  for idx, batch in enumerate(tqdm(train_dataloader)):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)
    attention_mask = batch.pop("attention_mask").to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids,
                    attention_mask = attention_mask)
    
    loss = outputs.loss

    #print("Loss:", loss.item())
    epoch_train_loss += (loss.item())

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

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids,
                        attention_mask = attention_mask)
        
        loss = outputs.loss

        #print("Valid Loss:", loss.item())  

        epoch_valid_loss += loss.item()

    print("Train Loss:",epoch_train_loss/len(train_dataloader.dataset))
    print("Valid Loss:",epoch_valid_loss/len(valid_dataloader.dataset))
    loss_df = loss_df.append({'epoch':epoch+1,'train_loss': epoch_train_loss/len(train_dataloader.dataset), 'valid_loss': epoch_valid_loss/len(valid_dataloader.dataset)}, ignore_index=True)

    valid_ave = epoch_valid_loss/len(valid_dataloader.dataset)
    if best_valid_loss >= valid_ave:
       best_valid_loss = valid_ave
       torch.save(model.state_dict(), weight_path)
loss_df.to_csv("/home/forte/users/nishida/loss.csv".format(dt_now))
loss_df.to_csv("/home/forte/users/nishida/loss/{}.csv".format(dt_now))


loss_csv = pd.read_csv("/home/forte/users/nishida/loss.csv")
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
plt.legend()

# Visualize and save
plt.savefig("/home/forte/users/nishida/graph/{}.png".format(dt_now))
plt.show()



model.load_state_dict(torch.load(weight_path))


model.eval()
test_loss = 0
with torch.no_grad():
   for i in tqdm(range(len(test_df_mscoco))):
    image_id = test_df_mscoco["path"][i]
    caption = test_df_mscoco["caption"][i]

    image = Image.open(image_id)
    encoding = processor(images=image, text=caption, padding="max_length", return_tensors="pt")        
    encoding = {k:v.squeeze() for k,v in encoding.items()}

    input_ids = (encoding["input_ids"]).to(device)
    pixel_values = (encoding["pixel_values"]).to(device)

    input_ids = input_ids.unsqueeze(dim=0)
    pixel_values = pixel_values.unsqueeze(dim=0)
    
    outputs = model(input_ids=input_ids,pixel_values=pixel_values,labels=input_ids)
    
    loss = outputs.loss
    test_loss += loss.item()

    ids = model.generate(pixel_values,max_length=50)
    generated = processor.batch_decode(ids,skip_special_token=True)[0]

    result_df = result_df.append({"image_id":image_id,"correct":caption,"generate":generated},ignore_index=True)
    #print(generated)

result_df.to_csv(result_path_mscoco,index=False)


with torch.no_grad():
   for i in tqdm(range(len(test_path_flickr))):
    image_id = test_df_flickr["path"][i]
    caption = test_df_flickr["caption"][i]

    image = Image.open(image_id)
    encoding = processor(images=image, text=caption, padding="max_length", return_tensors="pt")        
    encoding = {k:v.squeeze() for k,v in encoding.items()}

    input_ids = (encoding["input_ids"]).to(device)
    pixel_values = (encoding["pixel_values"]).to(device)

    input_ids = input_ids.unsqueeze(dim=0)
    pixel_values = pixel_values.unsqueeze(dim=0)
    
    outputs = model(input_ids=input_ids,pixel_values=pixel_values,labels=input_ids)
    
    loss = outputs.loss
    test_loss += loss.item()

    ids = model.generate(pixel_values,max_length=50)
    generated = processor.batch_decode(ids,skip_special_token=True)[0]

    result_df = result_df.append({"image_id":image_id,"correct":caption,"generate":generated},ignore_index=True)
    #print(generated)

result_df.to_csv(result_path_flickr,index=False)