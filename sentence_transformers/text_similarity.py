"""
https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
"""
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import heapq

sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]

model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = model.encode(sentences)



print(sentence_embeddings.shape)

print(f"target: {sentences[0]}")
# find similarities
res = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings
)
print(res)

# prepare result
def getTwo(fr: list) -> List[int]:
    # return the index number of the max and the second largest values
    return heapq.nlargest(2, range(len(fr)), key=fr.__getitem__)

# Result
max_indices = getTwo(res[0])
for i, index in enumerate(max_indices):
    label = sentences[index]
    print(f"{i}: {label}")

