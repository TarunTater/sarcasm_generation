##uses sense2vec
import os,pickle, spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 

lemmatizer = WordNetLemmatizer()
verb_forms= pickle.load(open("verb_data.pkl","rb"))
sid = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

def find_adjectives_adverbs(doc):
	adj_list = {}
	adv_list = {}
	for tok in doc:
		if tok.pos_=="ADJ":
			adj_list[tok.i] = tok.text
		elif tok.pos_=="ADV":
			adv_list[tok.i] = tok.text
		
	return [adj_list,adv_list]
	
def find_root(doc):
	for token in doc:
		if token.head == token:
			return token
	return None
def find_lemma(root):
	lemma = lemmatizer.lemmatize(root, pos="v")
	return lemma

def find_root_aspect(root):
	if root.tag_[-1]=="D":
		return "past"
	elif root.tag_[-1] == "G":
		return "present_participle"
	elif root.tag_[-1] == "N":
		return "past_participle"
	elif root.tag_[-1] == "Z":
		return "third_person_present"
	else:
		return "present_plural"

def find_target_verb_form(lemma,aspect):
	forms = verb_forms.get(lemma,[])
	#print (forms)
	if forms ==[]:
		return None
	if aspect == "third_person_present":
		return forms[0]
	elif aspect == "present_plural":
		return forms[1]
	elif aspect == "past":
		return forms[2]
	elif aspect=="present_participle":
		return forms[3]
	elif aspect=="past_participle":
		return forms[4]
	return None

def flip_root(root):
	lemma = find_lemma(root.text)
	aspect = find_root_aspect(root)
	#does the verb contain sentiment 
	pol = sid.polarity_scores(lemma)
	final_antonym = ""
	if pol['pos']>0 or pol['neg']>0:
		synsets = wordnet.synsets(lemma,pos="v")
		for syn in synsets:
			for l in syn.lemmas():
				if l.antonyms():
					antonym = l.antonyms()[0].name()
					pol1 = sid.polarity_scores(antonym)
					if pol1['pos']>0 or pol1['neg']>0:
						verb_form = find_target_verb_form(antonym,aspect)
						if verb_form!=None:
							return verb_form
	return None

def flip_adjectives_adverbs(adj_list,adv_list):
	new_list = {}
	for i in adj_list.keys():
		a = adj_list[i]
		pol = sid.polarity_scores(a)
		if pol['pos']>0 or pol['neg']>0:
			synsets = wordnet.synsets(a,pos="a")
			for syn in synsets:
				for l in syn.lemmas():
					if l.antonyms():
						antonym = l.antonyms()[0].name()
						pol1 = sid.polarity_scores(antonym)
						if pol1['pos']>0 or pol1['neg']>0:
							new_list[i] = antonym
							break
				else:
					continue
	for i in adv_list.keys():
		a = adv_list[i]
		pol = sid.polarity_scores(a)
		if pol['pos']>0 or pol['neg']>0:
			synsets = wordnet.synsets(a,pos="r")
			for syn in synsets:
				for l in syn.lemmas():
					if l.antonyms():
						antonym = l.antonyms()[0].name()
						pol1 = sid.polarity_scores(antonym)
						if pol1['pos']>0 or pol1['neg']>0:
							new_list[i] = antonym
							break
				else:
					continue
	return new_list
			
def construct_inverted_sentence(sentence):
	doc = nlp(sentence)
	a,r = find_adjectives_adverbs(doc)
	flipped_list = flip_adjectives_adverbs(a,r)
	new_sentence = "" #
	root = find_root(doc)
	if root!=None:
		flipped_root = flip_root(root)
	else:
		flipped_root = None					
	for tok in doc:
		if tok.i==root.i:
			if flipped_root!=None:
				new_sentence+=flipped_root+" "
			else:
				new_sentence+=root.text+" "
		else:
			if tok.i in flipped_list.keys():
				new_sentence+=flipped_list[tok.i]+" "
			else:
				new_sentence+=tok.text+" "
	return new_sentence.strip()
	

if __name__=="__main__":
	while (True):
		print ("Enter a sentence...")
		sentence = input()
		print (construct_inverted_sentence(sentence))
