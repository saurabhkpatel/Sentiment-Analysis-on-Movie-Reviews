'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads the first limit number of ham emails and the first limit number of spam.
  It creates an "emaildocs" variable with a list of emails consisting of a pair
    with the list of tokenized words from the email and the label either spam or ham.
  It prints a few example emails.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySPAM  <corpus directory path> <limit number>
'''

# open python and nltk packages needed for processing
import sentiment_read_subjectivity
import os
import sys
import random
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.metrics import ConfusionMatrix
import re
#import sentiment_read_subjectivity

stopwords = nltk.corpus.stopwords.words('english')
newstopwords = [word for word in stopwords if word not in ['not', 'no', 'can','don', 't']]


#import sentiment__read_subjectivity_words
# initialize the positive, neutral and negative word lists
#(positivelist, neutrallist, negativelist) = #sentiment__read_subjectivity_words.read_three_types()

#########################################################################################
####  Pre-processing the documents  ####
#########################################################################################
def pre_processing_documents(document):
  # "Pre_processing_documents"  
  # "create list of lower case words"
  word_list = re.split('\s+', document.lower())
  # punctuation and numbers to be removed
  punctuation = re.compile(r'[-.?!/\%@,":;()|0-9]')
  word_list = [punctuation.sub("", word) for word in word_list] 
  final_word_list = []
  for word in word_list:
    if word not in newstopwords:
      final_word_list.append(word)
  line = " ".join(final_word_list)
  return line 

#########################################################################################
####  get words/tokens from documents  ####
#########################################################################################
def get_words_from_phasedocs(docs):
  all_words = []
  for (words, sentiment) in docs:
    # more than 3 length
    possible_words = [x for x in words if len(x) >= 3]
    all_words.extend(possible_words)
  return all_words

def get_words_from_phasedocs_normal(docs):
  all_words = []
  for (words, sentiment) in docs:
    all_words.extend(words)
  return all_words  

# get all words from tokens
def get_words_from_test(lines):
  all_words = []
  for id,words in lines:
    all_words.extend(words)
  return all_words

#########################################################################################
####  Function writeFeatureSets to csv file ####
#########################################################################################
# Function writeFeatureSets:
# takes featuresets defined in the nltk and convert them to weka input csv file
#    any feature value in the featuresets should not contain ",", "'" or " itself
#    and write the file to the outpath location
#    outpath should include the name of the csv file
def writeFeatureSets(featuresets, outpath):
    # open outpath for writing
    f = open(outpath, 'w')
    # get the feature names from the feature dictionary in the first featureset
    featurenames = featuresets[0][0].keys()
    # create the first line of the file as comma separated feature names
    #    with the word class as the last feature name
    featurenameline = ''
    for featurename in featurenames:
        # replace forbidden characters with text abbreviations
        featurename = featurename.replace(',','CM')
        featurename = featurename.replace("'","DQ")
        featurename = featurename.replace('"','QU')
        featurenameline += featurename + ','
    featurenameline += 'class'
    # write this as the first line in the csv file
    f.write(featurenameline)
    f.write('\n')
    # convert each feature set to a line in the file with comma separated feature values,
    # each feature value is converted to a string 
    #   for booleans this is the words true and false
    #   for numbers, this is the string with the number
    for featureset in featuresets:
        featureline = ''
        for key in featurenames:
            featureline += str(featureset[0][key]) + ','
        if featureset[1] == 0:
          featureline += str("neg")
        elif featureset[1] == 1:
          featureline += str("sneg")
        elif featureset[1] == 2:
          featureline += str("neu")
        elif featureset[1] == 3:
          featureline += str("spos")
        elif featureset[1] == 4:
          featureline += str("pos")
        # write each feature set values to the file
        f.write(featureline)
        f.write('\n')
    f.close()

# define a feature definition function here
####################################################################
#### Get word-features, bag of words  ####
####################################################################
def get_word_features(wordlist):
  wordlist = nltk.FreqDist(wordlist)
  word_features = [w for (w, c) in wordlist.most_common(200)] 
  return word_features    

# define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'contains(keyword)' and is true or false depending
# on whether that keyword is in the document
# define a feature definition function here
def normal_features(document, word_features):
  document_words = set(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  return features


# define a feature definition function here
####################################################################
#### Get bigram document features  ####
####################################################################
# define features that include words as before 
# add the most frequent significant bigrams
# this function takes the list of words in a document as an argument and returns a feature dictionary
# it depends on the variables word_features and bigram_features
def bigram_document_features(document, word_features,bigram_features):
  document_words = set(document)
  document_bigrams = nltk.bigrams(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  for bigram in bigram_features:
    features['bigram({} {})'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
  return features
def get_biagram_features(tokens):
  bigram_measures = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(tokens,window_size=3)
  #finder.apply_freq_filter(6)
  bigram_features = finder.nbest(bigram_measures.chi_sq, 3000)
  return bigram_features[:500]

####################################################################
#### Sentiment Lexicon: Subjectivity Count features  ####
####################################################################
# define features that include word counts of subjectivity words
# negative feature will have number of weakly negative words +
#    2 * number of strongly negative words
# positive feature has similar definition
#    not counting neutral words
# create your own path to the subjclues file
def readSubjectivity(path):
  flexicon = open(path, 'r')
  # initialize an empty dictionary
  sldict = { }
  for line in flexicon:
    fields = line.split()   # default is to split on whitespace
    # split each field on the '=' and keep the second part as the value
    strength = fields[0].split("=")[1]
    word = fields[2].split("=")[1]
    posTag = fields[3].split("=")[1]
    stemmed = fields[4].split("=")[1]
    polarity = fields[5].split("=")[1]
    if (stemmed == 'y'):
      isStemmed = True
    else:
      isStemmed = False
    # put a dictionary entry with the word as the keyword
    #     and a list of the other values
    sldict[word] = [strength, posTag, isStemmed, polarity]
  return sldict

SLpath = "./SentimentLexicons/subjclueslen1-HLTEMNLP05.tff"
SL = readSubjectivity(SLpath)
def SL_features(document, word_features, SL):
  document_words = set(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  # count variables for the 4 classes of subjectivity
  weakPos = 0
  strongPos = 0
  weakNeg = 0
  strongNeg = 0
  for word in document_words:
    if word in SL:
      strength, posTag, isStemmed, polarity = SL[word]
      if strength == 'weaksubj' and polarity == 'positive':
        weakPos += 1
      if strength == 'strongsubj' and polarity == 'positive':
        strongPos += 1
      if strength == 'weaksubj' and polarity == 'negative':
        weakNeg += 1
      if strength == 'strongsubj' and polarity == 'negative':
        strongNeg += 1
      features['positivecount'] = weakPos + (2 * strongPos)
      features['negativecount'] = weakNeg + (2 * strongNeg)
  
  if 'positivecount' not in features:
    features['positivecount']=0
  if 'negativecount' not in features:
    features['negativecount']=0      
  return features

###################################################
####   Negation words features  ####
####################################################
# Negation words "not", "never" and "no"
# Not can appear in contractions of the form "doesn", "'", "t"
## if', 'you', 'don', "'", 't', 'like', 'this', 'film', ',', 'then', 'you', 'have', 'a', 'problem', 'with', 'the', 'genre', 'itself', 
# One strategy with negation words is to negate the word following the negation word
#   other strategies negate all words up to the next punctuation
# Strategy is to go through the document words in order adding the word features,
#   but if the word follows a negation words, change the feature to negated word
# Start the feature set with all 2000 word features and 2000 Not word features set to false
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
def NOT_features(document, word_features, negationwords):
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = False
    features['contains(NOT{})'.format(word)] = False
  # go through document words in order
  for i in range(0, len(document)):
    word = document[i]
    if ((i + 1) < len(document)) and (word in negationwords):
      i += 1
      features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
    else:
      if ((i + 3) < len(document)) and (word.endswith('n') and document[i+1] == "'" and document[i+2] == 't'):
        i += 3
        features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
      else:
        features['contains({})'.format(word)] = (word in word_features)
  return features

###################################################
####   POS tag features  ####
####################################################
# this function takes a document list of words and returns a feature dictionay
#   it depends on the variable word_features
# it runs the default pos tagger (the Stanford tagger) on the document
#   and counts 4 types of pos tags to use as features
def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features

###################################################
####   processkaggle data  ####
####################################################
# function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('./train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

  #for phrase in phraselist[:10]:
    #print (phrase)
  
  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  phrasedocs_without = []
  # add all the phrases
  for phrase in phraselist:

    #without preprocessing
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs_without.append((tokens, int(phrase[1])))
    
    # with pre processing
    tokenizer = RegexpTokenizer(r'\w+')
    phrase[0] = pre_processing_documents(phrase[0])
    tokens = tokenizer.tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))
  
  # possibly filter tokens
  normaltokens = get_words_from_phasedocs_normal(phrasedocs_without)
  preprocessedTokens = get_words_from_phasedocs(phrasedocs)

  # continue as usual to get all words and create word features


    
  word_features = get_word_features(normaltokens)
  featuresets_without_preprocessing = [(normal_features(d, word_features), s) for (d, s) in phrasedocs_without]
  #print featuresets_without_preprocessing[0]
  writeFeatureSets(featuresets_without_preprocessing,"features_normal.csv")
  print "---------------------------------------------------"
  print "Accuracy with normal features, without pre-processing steps : "
  accuracy_calculation(featuresets_without_preprocessing)


  word_features = get_word_features(preprocessedTokens)
  #print word_features[:20]
  featuresets = [(normal_features(d, word_features), s) for (d, s) in phrasedocs]
  #print featuresets[0]
  writeFeatureSets(featuresets,"features_preprocessed.csv")
  print "---------------------------------------------------"
  print "Accuracy with pre-processed features : "
  accuracy_calculation(featuresets)

  
  SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in phrasedocs]
  writeFeatureSets(SL_featuresets,"features_SL.csv")
  #print SL_featuresets[0]
  print "---------------------------------------------------"
  print "Accuracy with SL_featuresets : "
  accuracy_calculation(SL_featuresets)

  NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in phrasedocs]
  #print NOT_featuresets[0]
  writeFeatureSets(SL_featuresets,"features_NOT.csv")
  print "---------------------------------------------------"
  print "Accuracy with NOT_featuresets : "
  accuracy_calculation(NOT_featuresets)

  bigram_features = get_biagram_features(preprocessedTokens)
  #print bigram_features[0]
  bigram_featuresets = [(bigram_document_features(d, word_features,bigram_features), c) for (d, c) in phrasedocs]
  #print bigram_featuresets[0]
  writeFeatureSets(bigram_featuresets,"features_biagram.csv")
  print "---------------------------------------------------"
  print "Accuracy with bigram featuresets : "
  accuracy_calculation(bigram_featuresets)


  '''pos_featuresets = [(POS_features(d, word_features), c) for (d, c) in phrasedocs]
  print pos_featuresets[0]
  writeFeatureSets(pos_featuresets,"features_pos.csv")
  print "---------------------------------------------------"
  print "Accuracy with pos_featuresets : "
  writeFeatureSets(pos_featuresets,"features_pos_featuresets.csv")
  accuracy_calculation(pos_featuresets)'''



###################################################
####    generate test csv file. ####
####################################################
  '''
  f = open('./test.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  testphrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      tempList = []
      if len(line.split('\t')) == 3:
        tempList.append(line.split('\t')[0])
        tempList.append(line.split('\t')[2])
        testphrasedata.append(tempList)
      else :
        tempList.append(line.split('\t')[0])
        tempList.append("")
        testphrasedata.append(tempList)
  
  phraselist = testphrasedata
  print('Read', len(testphrasedata), 'phrases, using', len(phraselist), 'test phrases')

  #for phrase in phraselist[:10]:
    #print (phrase)

  # create list of phrase documents as (list of words)
  phrasedocs = []
  # add all the phrases
  for id,phrase in phraselist:
    # with pre processing
    tokenizer = RegexpTokenizer(r'\w+')
    phrase = pre_processing_documents(phrase)
    tokens = tokenizer.tokenize(phrase)
    phrasedocs.append((id,tokens))

  preprocessedTestTokens = get_words_from_test(phrasedocs)
  test_word_features = get_word_features(preprocessedTestTokens)
  
  test_featuresets = [(normal_features(d, test_word_features), id) for (id, d) in phrasedocs]
  create_test_submission(featuresets,test_featuresets,"sample.csv")'''

  
#########################################################################################
####  Features   accuracy calculation####
#########################################################################################
def accuracy_calculation(featuresets):
  print "Training and testing a classifier "  
  training_size = int(0.1*len(featuresets))
  test_set = featuresets[:training_size]
  training_set = featuresets[training_size:]
  classifier = nltk.NaiveBayesClassifier.train(training_set)
  print "Accuracy of classifier : "
  print nltk.classify.accuracy(classifier, test_set)
  print "---------------------------------------------------"
  print "Showing most informative features"
  print classifier.show_most_informative_features(30)
  #print "---------------------------------------------------"
  #print "precision, recall and F-measure scores"
  print_confusionmatrix(classifier,test_set)
  print ""  
  
#########################################################################################
## print ConfusionMatrix. ##
#########################################################################################
def print_confusionmatrix(classifier_type, test_set):
  reflist = []
  testlist = []
  for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier_type.classify(features))
  
  print " "
  print "The confusion matrix"
  cm = ConfusionMatrix(reflist, testlist)
  print cm

  
#########################################################################################
####  create test submission file, predict label using classifier ####
#########################################################################################
def create_test_submission(featuresets,test_featuresets,fileName):
  print "---------------------------------------------------"
  print "Training and testing a classifier "  
  test_set = test_featuresets
  training_set = featuresets
  classifier = nltk.NaiveBayesClassifier.train(training_set)

  fw = open(fileName,"w")
  fw.write("PhraseId"+','+"Sentiment"+'\n')
  for test,id in test_featuresets:
    fw.write(str(id)+','+str(classifier.classify(test))+'\n')
  fw.close()

"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.
"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])

