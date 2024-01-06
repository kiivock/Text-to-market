import nltk

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

def remove_numbers(text):
    text=str(text)
    text_clean=''.join(char for char in text if not char.isdigit())
    return text_clean

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens_cleaned = [w for w in tokens if not w in stop_words]
    text_clean=''
    for w in tokens_cleaned:
        text_clean=' '.join(tokens_cleaned)
    return text_clean

def lemmatize_text(text):
    tokens = word_tokenize(text)
    verb_lemmatized = [
                        WordNetLemmatizer().lemmatize(word, pos = "v")
                        for word in tokens
                    ]
    noun_lemmatized = [
                        WordNetLemmatizer().lemmatize(word, pos = "n")
                        for word in verb_lemmatized
                        ]

    text_clean=' '.join(noun_lemmatized)
    return text_clean

def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    #drop date column
    data=df.drop(["Date"],axis=1)

    # stripping
    data=data.apply(lambda x: x.str.strip())
    data=data.apply(lambda x: x.str.strip("b"))

    #lowercase
    data=data.apply(lambda x: x.str.lower())

    #removing punctuation
    for punctuation in string.punctuation:
        data=data.apply(lambda x: x.str.replace(punctuation,""))

    #removing numbers
    for column in data.columns:
        data[column]=data[column].apply(remove_numbers)

    #remove stopwords
    for column in data.columns:
        data[column]=data[column].apply(remove_stopwords)

    # lemmatize
    for column in data.columns:
        data[column]=data[column].apply(lemmatize_text)

    #take back the date
    data['Date']=df['Date']
    cols=data.columns.tolist()
    cols=cols[-1:] + cols[:-1]
    data=data[cols]

    print ("✅ preprocess_text() done \n")

    return data
