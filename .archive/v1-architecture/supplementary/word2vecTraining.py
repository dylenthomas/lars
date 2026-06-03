import json
from gensim.models import Word2Vec

### Open and process the whisper corpus ----
with open("WhisperCorpus.json", "r") as jsn_data:
    corpus = jsn_data.read()
    jsn_data.close()

# remove the first colon
corpus = corpus.split(',')
corpus[0] = corpus[0].replace("{", "")

# remove Gs and indexes
for i, samp in enumerate(corpus):
    if "Ġ" in samp: samp = samp.replace("Ġ", "")
    samp = samp.replace('"', "")
    colon = samp.find(":")
    samp = samp[0:colon]
    corpus[i] = samp

# format for training
corpus = [[x] for x in corpus]

# rejoin into a string then write into a text file
#corpus = ", ".join(corpus)

#with open("processedWhisperCorpus.txt", "w") as processed:
#    processed.write(corpus)
#    processed.close()

model = Word2Vec(sentences=corpus, vector_size=25, window=5, min_count=1, workers=4)
model.train(corpus, total_examples=1, epochs=100)
model = model.wv

print(model['lights', 'on'])
print(model['lights', 'on'].shape)
print(model['phono'])
print(model['phono'].shape)

model.save("../.model/LARS.wordvectors")