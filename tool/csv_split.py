#データセット(csv)を分割
#必要

import pandas as pd

all_df = pd.read_csv("/home/forte/users/nishida/dataset/vqa/vqa_flickr_tifa.csv")
print(len(all_df))

seed = int(0.8*len(all_df))
print(seed)

train_df = all_df[:seed]
test_df = all_df[seed:]
print(len(train_df))
print(len(test_df))


train_df.to_csv("/home/forte/users/nishida/dataset/vqa/vqa_flikcrtifa_train.csv",index=False)
test_df.to_csv("/home/forte/users/nishida/dataset/vqa/vqa_flikcrtifa_test.csv",index=False)