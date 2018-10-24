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

###################################################################


#More regex with re.search()
#In this exercise, you'll utilize re.search() and re.match() to find specific tokens. 
#Both search and match expect regex patterns, similar to those you defined in an earlier exercise. 
#You'll apply these regex library methods to the same Monty Python text from the nltk corpora.

#You have both scene_one and sentences available from the last exercise; now you can use them with 
#re.search() and re.match() to extract and match more text.

# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())

# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))

# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))






