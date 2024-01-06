import pandas as pd
from ml_logic.preprocessing_text import preprocess_text

df = pd.read_csv("/Users/yassir2/code/Yassirbenj/text_to_market/raw_data/Combined_News_DJIA.csv")
data=df.drop(["Label"],axis=1)

data_preprocess=preprocess_text(data)

#take back label
data_preprocess['Label']=df['Label']

#export to csv
data_preprocess.to_csv("/Users/yassir2/code/Yassirbenj/text_to_market/processed_data/preprocess_text.csv")
