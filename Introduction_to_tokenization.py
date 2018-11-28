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


####################################################################################

                          #Choosing a tokenizer
#Given the following string, which of the below patterns is the best tokenizer? 
#If possible, you want to retain sentence punctuation as separate tokens, but have '#1' remain a single token.

#my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"
#The string is available in your workspace as my_string, and the patterns have been pre-loaded as pattern1, 
#pattern2, pattern3, and pattern4, respectively.

#Additionally, regexp_tokenize has been imported from nltk.tokenize. You can use regexp_tokenize() 
#with my_string and one of the patterns as arguments to experiment for yourself and see which is the best tokenizer.

#Answer would be : r"(\w+|#\d|\?|!)"

###############################################
                          
                      #Regex with NLTK tokenization
#Twitter is a frequently used source for NLP text and tasks. In this exercise, 
#you'll build a more complex tokenizer for tweets with hashtags and mentions using nltk and regex. 
#The nltk.tokenize.TweetTokenizer class gives you some extra methods and attributes for parsing tweets.

#Here, you're given some example tweets to parse using both TweetTokenizer and regexp_tokenize from 
#the nltk.tokenize module. These example tweets have been pre-loaded into the variable tweets. 
#Feel free to explore it in the IPython Shell!

# Import the necessary modules
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import TweetTokenizer

# Define a regex pattern to find hashtags: pattern1 
pattern1 = r"#\w+"

# Use the pattern on the first tweet in the tweets list
regexp_tokenize(tweets[0], pattern1)

# Write a pattern that matches both mentions and hashtags
pattern2 = r"([#|@]\w+)"

# Use the pattern on the last tweet in the tweets list
regexp_tokenize(tweets[-1], pattern2)

# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)

##############################################################

                                  #Non-ascii tokenization
#In this exercise, you'll practice advanced tokenization by tokenizing some non-ascii based text. You'll be using German with emoji!

#Here, you have access to a string called german_text, which has been printed for you in the Shell. 
#Notice the emoji and the German characters!

#The following modules have been pre-imported from nltk.tokenize: regexp_tokenize and word_tokenize.

#Unicode ranges for emoji are:

#('\U0001F300'-'\U0001F5FF'), ('\U0001F600-\U0001F64F'), ('\U0001F680-\U0001F6FF'), and ('\u2600'-\u26FF-\u2700-\u27BF').

# Tokenize and print all words in german_text
all_words = word_tokenize(german_text)
print(all_words)

# Tokenize and print only capital words
capital_words = r"[A-ZÜ]\w+"
print(regexp_tokenize(german_text, capital_words))

# Tokenize and print only emoji
emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print(regexp_tokenize(german_text, emoji))

####################################################################################################################

                      #Charting practice
#Try using your new skills to find and chart the number of words per line in the script using matplotlib. 
#The Holy Grail script is loaded for you, and you need to use regex to find the words per line.

#Using list comprehensions here will speed up your computations. 
#For example: my_lines = [tokenize(l) for l in lines] will call a function tokenize on each line in the list lines. 
#The new transformed list will be saved in the my_lines variable.

#You have access to the entire script in the variable holy_grail. Go for it!

with open('grail.txt', 'r') as content_file:
    holy_grail = content_file.read()
    
type(holy_grail)
holy_grail

# Split the script into lines: lines
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, '', l) for l in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(s,r"\w+") for s in lines]

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t_line) for t_line in tokenized_lines]

# Plot a histogram of the line lengths
plt.hist(line_num_words)

# Show the plot
plt.show()








