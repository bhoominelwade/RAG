import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore") #removing output noise


from sentence_transformers import SentenceTransformer,util #convert to vectors, built in cosine lib
import numpy as np #make a function to calculate cosine similarity search
from dotenv import load_dotenv #load apis
from anthropic import Anthropic


model = SentenceTransformer('all-MiniLM-L6-v2')
load_dotenv()
client = Anthropic() #retrives api from env
results = [] 


text = "weather today is hot. It is not raining in Mumbai . I like winters over summers and rain.I love to go outside. It is a cool day in Mumbai"
chunks = text.split(".")
chunks_embedding = model.encode(chunks)


query = input("Ask a question: ")
query_embedding = model.encode(query)


for i in range(len(chunks_embedding)):
    score = util.cos_sim(chunks_embedding[i], query_embedding)
    results.append([score, chunks[i]])
    
results.sort(reverse = True)
top_chunks = [chunk for score, chunk in results[:2]] 

context = "\n".join(top_chunks)
prompt = f"""Answer the question using only the context below.
Context:{context}
Question: {query}
Answer:
"""

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens = 200,
    messages = [{'role':'user','content':prompt}]
)

print(response.content[0].text)
    

