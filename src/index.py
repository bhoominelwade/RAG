import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore") #removing output noise

# this file is to create and store embeddings and chunks for refernce

from sentence_transformers import SentenceTransformer,util #convert to vectors, built in cosine lib
import numpy as np #make a function to calculate cosine similarity search

model = SentenceTransformer('all-MiniLM-L6-v2')

text = "weather today is hot. It is not raining in Mumbai . I like winters over summers and rain.I love to go outside. It is a cool day in Mumbai"
chunks = text.split(".")
chunks_embedding = model.encode(chunks)

# saving into files
np.save("data/embedding.npy",chunks_embedding)
with open("data/chunks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(chunks))