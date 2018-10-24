#Word tokenization with NLTK
#Here, you'll be using the first scene of Monty Python's Holy Grail, which has been pre-loaded as scene_one. 
#Feel free to check it out in the IPython Shell!

#Your job in this exercise is to utilize word_tokenize and sent_tokenize from nltk.tokenize to tokenize both 
#words and sentences from Python strings - in this case, the first scene of Monty Python's Holy Grail.


# Import necessary modules
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)


