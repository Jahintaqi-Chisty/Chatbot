from django.shortcuts import render
import nltk
import random
import string
import re, string, unicodedata
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import wikipedia as wk
from collections import defaultdict
import warnings
import json
from django.http import HttpResponse
from django.http import JsonResponse

warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

welcome_input = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
welcome_response = ["hi", "hey","hi there", "hello", "I am glad! You are talking to me"]




# Create your views here.

def temp(request):
    # user_response = request.POST['text']


    if request.is_ajax():
        if request.method == 'POST':
            response = json.loads(request.body.decode('utf-8'))
            t = chat(response)
            # print(response['text'])
            # t = json.loads('{"message":"Jahin"}')
            return JsonResponse({"message": t})
            # return HttpResponse(t)
    elif request.method == 'GET':
        return render(request, 'jsapp.html')


def welcome(user_response):
    for word in user_response.split():
        if word.lower() in welcome_input:
            return random.choice(welcome_response)


def wikipedia_data(input):
    reg_ex = re.search('tell me about (.*)', input)
    try:
        if reg_ex:
            topic = reg_ex.group(1)
            wiki = wk.summary(topic, sentences=3)
            return wiki
    except Exception as e:
        text = "No content has been found"
        return text


def Normalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    # word tokenization
    word_token = nltk.word_tokenize(text.lower().translate(remove_punct_dict))

    # remove ascii
    new_words = []
    for word in word_token:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)

    # Remove tags
    rmv = []
    for w in new_words:
        text = re.sub("&lt;/?.*?&gt;", "&lt;&gt;", w)
        rmv.append(text)

    # pos tagging and lemmatization
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    rmv = [i for i in rmv if i]
    for token, tag in nltk.pos_tag(rmv):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list


def generateResponse(user_response):
    # data = open('data/HR.txt', 'r', errors='ignore')
    data = open('data/diu_tution.txt', 'r', errors='ignore')
    raw = data.read()
    raw = raw.lower()
    Normalize(raw)
    sent_tokens = nltk.sent_tokenize(raw)
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # vals = cosine_similarity(tfidf[-1], tfidf)
    vals = linear_kernel(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0) or "tell me about" in user_response:
    # if "tell me about" in user_response:
        # print("Checking Wikipedia")
        if user_response:
            robo_response = wikipedia_data(user_response)
            return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response#wikipedia search


def chat(text):
    # Text from user

    # initial function
    flag = True
    while flag == True:
        # user_response = request.POST['text']
        user_response = text['text']
        user_response = user_response.lower()
        if user_response not in ['bye', 'shutdown', 'exit', 'quit']:
            if user_response == 'thanks' or user_response == 'thank you':
                flag = False
                text = "Chatterbot : You are welcome.."
            else:
                if welcome(user_response) is not None:
                    flag = False
                    text = str(( welcome(user_response)))
                else:
                    # print("Chatterbot : ", end="")
                    text = (generateResponse(user_response))
                    # sent_tokens.remove(user_response)
                    break

        else:
            flag = False
            text = "Bye!!!"

    return text
