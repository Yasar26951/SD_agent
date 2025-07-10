import os
import pandas as pd
import tensorflow as tf

from transformers import BertTokenizer, TFBertModel, AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import numpy as np
from . import demo

from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.mixed_precision import Policy as mp_policy
import os
print("Current working directory:", os.getcwd())
print("Does file exist?", os.path.exists("model2/bert11.h5"))


['bert', 'lstm', 'gru', 'comment', 'email', 'message', 'tweet']
import numpy as np
from mistralai import Mistral
database=demo.database()
from dotenv import load_dotenv
load_dotenv()
class master:
    def __init__(self):
            api_key =os.getenv("MasterApi")
            self.client = Mistral(api_key=api_key)
    def check(self, text,plat):
        prompt = (
            f"Platform: {plat}. Classify the input text as spam or not.\n"
            f"Respond with only 0 (not spam) or 1 (spam).\n"
            f"Text: '{text}'"
        )

        response = self.client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()

        return response.choices[0].message.content


    def data_check(self):
         global database
         da=database.show()
         response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user",
                       "content": f"i m building spam detector i need more  data i have only this [('text','comment','email ','message','tweet','spam'),(' random real world  message from social media ',1,0,0,0,1)] numpy format string please generate similar data  100 rows of data include  random text data  numpy format which i given. i need ony data ,dont give code.no code, include this {da} data also"}])
         d = response.choices[0].message.content
         print("master code")
         daat = d
         star = daat.index('[')
         end = len(daat) - daat[::-1].index("]")
         daat = daat[star+1:end-1]
         print(daat)
         da = daat.split(",\n")
         columns = ['x', 'comment', 'email', 'message','tweet','spam' ]
         import pandas as pd
         data=pd.DataFrame()
         data[columns[0]] = [str(j[2:-2].split(",")[1]) for j in da[1:]]
         print(da[4])
         for i in range(1,len(columns)):
             print([eval(j)[i] for j in da[1:]])
             data[columns[i]] = [eval(j)[i+1] for j in da[1:]]

       # data['spam'].transform(round)
         print(data)
         return data

    def reason(self,spam,text='message'):
        if round(spam)==1:
            sp='spam'
        else:
            sp='not spam '

        response = self.client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": f"""it came from {plat}. 
        tell me the reason why it is {sp} your assumption text is given below
        Text: {text}"""
                }
            ]
        )

        return response.choices[0].message.content.strip()

        d=response.choices[0].message.content
        print(d)
        return d





maste=master()

tf.keras.mixed_precision.set_global_policy('mixed_float16')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", vocab_size=10000)
with custom_object_scope({'DTypePolicy': mp_policy}):
    bert = tf.keras.models.load_model(r"D:/srm/SD/app/model2/bert11.h5", compile=False)
bert.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
def load_model(path):
    model = tf.keras.models.load_model(path,compile=False)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model
print("load the model")
lstm = load_model(r"model2/lstm11 (3).h5")
gru = load_model(r"model2/gru11 (3).h5")

engine = load_model(r"model2/engine11.h5")
print("load the done")


#bert using transformer


def bert1(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    text = list(text)
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
    cls_embedding = last_hidden_state[:, 0, :]
    return cls_embedding


#prepross the text


def preprocess(text):
    global tokenizer
    text = [text]
    print(text)
    stopwor = set(stopwords.words("english"))
    text = [re.sub(r'\b\d+(\\d+)?\b', '', t) for t in text]
    text = [word_tokenize(t) for t in text]
    text = [[i for i in t if i not in stopwor] for t in text]
    stemmer = PorterStemmer()
    text = [[stemmer.stem(i) for i in t] for t in text]
    text = [" ".join(t) for t in text]
    padded = tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors='tf')
    return np.array(padded['input_ids'])

def train():
    global lstm,gru,bert,engine
    data=maste.data_check()
    # if data.shape[0]<20:
    #     print(f"low number of data not able to train (least 20) but{data.shape[0]}")
    #     return 0
    msgs=data["x"]
    spam=data["spam"]
    spam.transform(round)
    padded=preprocess(msgs)
    lstm = load_model(r"model2/lstm11 (3).h5")
    gru = load_model(r"model2/gru11 (3).h5")
    bert = load_model(r"model2/bert11.h5")
    engine = load_model(r"model2/engine11.h5")
    print("Padded shape:", padded)
    print("Spam shape:",spam)
    gru.fit(padded.reshape(-1, 50), np.array(spam).reshape(-1, 1), batch_size=16, epochs=1)
    lstm.fit(padded.reshape(-1,50),np.array(spam).reshape(-1,1),batch_size=16,epochs=1)
    print("done")
    txt=bert1(msgs)
    bert.fit(np.array(txt).reshape(-1,768),np.array(spam).reshape(-1,1),batch_size=16,epochs=1)
    ls=lstm.predict(padded.reshape(-1, 50))
    gr=gru.predict(padded.reshape(-1, 50))
    be=bert.predict(np.array(txt).reshape(-1,768))
    dat=pd.DataFrame()

    dat["lstm"]=[i[0] for i in ls]
    dat["gru"] =[i[0] for i in gr]
    dat["bert"] =[i[0] for i in be]
    for i in ['comment','email','message','tweet']:
        dat[i]=data[i]
    x=dat.to_numpy()
    #print(x.shape)
    y=data["spam"].to_numpy()
    print(f"x shape: {x.shape}, dtype: {x.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    print(f"x sample: {x[0]}")
    print(f"y sample: {y[0]}")
    engine.fit(x.reshape(-1, 7),y.reshape(-1,1),batch_size=16, epochs=1)
    a = engine.predict(x.reshape(-1, 7))
    print(np.sum(((np.round(a)-np.round(y))**2)**0.5)/len(y))
    lstm.save(r"model2/lstm11 (3).h5")
    gru.save(r"model2/gru11 (3).h5")
    bert.save(r"model2/bert11.h5")
    engine.save(r"model2/engine11.h5")


    database.coppy()



#store the data
def store(sec,text,plat):
    global database
    print(plat)
    mas=maste.check(text,plat)
    sec[-1]=round(0.20*float(sec[-1])+0.8*float(mas))

    sec.pop(0)
    sec[0]=text
    sec.pop(1)
    print(sec)
    a=database.store(sec)
    print(a)






#predict the input

def predictting(text,plat):
    global lstm, engine, gru,bert
    print(text)
    encode = bert1(text)
    padded = preprocess(text)
    print(padded.shape)
    ls = lstm.predict(np.array(padded).reshape(1, -1) )
    print(1,ls.shape)
    gu = gru.predict(np.array(padded).reshape(1, -1) )
    print(2,gu.shape)
    ber=bert.predict(np.array(encode).reshape(-1,768))
    print(3)


    li=['comment', 'email', 'message', 'tweet']
    i=li.index(plat)
    pla=[0,0,0,0]
    for j in range(len(li)):
        if  j==i:
            pla[j]=1
    ls=[i[0] for i in ls]
    gu = [i[0] for i in gu]
    sec = np.array([sum(ls)/len(ls), sum(gu)/len(gu),ber[0][0],*pla])
    spam=engine.predict(sec.reshape(-1, 7))
    print(spam)
    sec=list(sec)
    sec.append(spam)
    print("ew",sec)

    print(plat)
    store(sec,text,plat)

    return spam



#train()
#predictting(' Win a free iPhone by clicking here: [Link]',"message")


