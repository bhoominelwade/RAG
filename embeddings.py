import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore") #removing output noise


from sentence_transformers import SentenceTransformer #convert to vectors
import numpy as np #make a function to calculate cosine similarity search

model = SentenceTransformer('all-MiniLM-L6-v2')

sentance = ["Today the weather is hot","its not raining in mumbai","I like winters"]
sentance_embedding = model.encode(sentance)

query = "How is the weather in mumbai"
query_embedding = model.encode(query)

def cosine(a,b): #function to calculate cosine similarity
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i in range(len(sentance_embedding)):
    score = cosine(sentance_embedding[i], query_embedding)
    print(f"{sentance[i]} => score {score}")
    
#for one vectore comparison   
# print(cosine(sentace_embedding, query_embedding)) 

