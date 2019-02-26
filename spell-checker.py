from math import log10, log, sqrt
import xml.etree.cElementTree as ET  # to read xml files
import re  # regular expressions
import sys
import codecs
import platform
import arpa
import os
import subprocess
import time
from joblib import Parallel, delayed
import multiprocessing
import copy
import numpy as np
import matplotlib.pyplot as plt
import string

import argparse


VERSION = "1.0.0"
SMOOTH_LM = 1e-06
SMOOTH_EM = 0.0001
PRIOR_WEIGHT =  1.25

# supports N = {1, 2 ,3}
N_GRAM = 1
DISTANCE_LIMIT = 2
COUNT_THRESHOLD = 2
#RETRAIN_ITERATIONS = 0

OOV = 13250000000 

OVERWRITE = False
DEST = ""
OCR = True
      



" Caching System "

def memo(f):
	"""Memoize function f."""

	table = {}

	def fmemo(*args):
		try:
			args[0]  # throws Exception
			if args not in fmemo.cache:
				fmemo.cache[args] = f(*args)
			return fmemo.cache[args]
		except IndexError:
			fmemo.cache = {}

	fmemo.cache = table
	return fmemo


def memo2(f):
	"""Memoize function f."""

	table = {}

	def fmemo(*args):
		try:
			args[1]  # throws Exception
			if args[1:] not in fmemo.cache:
				fmemo.cache[args[1:]] = f(*args)
			return fmemo.cache[args[1:]]
		except IndexError:
			fmemo.cache = {}

	fmemo.cache = table
	return fmemo

def memo3(f):
	"""Memoize function f."""

	table = {}

	def fmemo(*args):
		try:
			args[1]  # throws Exception
			if args[2:] not in fmemo.cache:
				fmemo.cache[args[2:]] = f(*args)
			return fmemo.cache[args[2:]]
		except IndexError:
			fmemo.cache = {}

	fmemo.cache = table
	return fmemo


PREFIXES = []
blackList = {}


class LanguageModelOLD(dict):
	
	Vocabulary = {}
	N = 0
	
	def __init__(self, newInstance):
		
		if not newInstance:
			for line in open("/nethome/cklaus/Desktop/BachelorThesis/LanguageModel.count", "r", encoding="iso-8859-1"):
				line = line.split()

				#self.setProperty( " ".join(line[0:-1]) , int(line[-1]))
				self[" ".join(line[0:-1])] = int(line[-1])
				if len(line) == 2:
					self.N += int(line[-1])

	@memo2
	def __call__(self, current, prev2=None, prev1=None):
		
		global SMOOTH_LM

		'Unigrams'
		if prev2 is None:
			return log10((self.get(current, 0) + SMOOTH_LM) / float(self.N + SMOOTH_LM * len(self.Vocabulary)))

			'Bigrams'
		elif prev1 is None:
			if prev2+" "+current in self:
				return log10((self.get(prev2+" "+current, 0) + SMOOTH_LM) / float(self.get(prev2, 0) + SMOOTH_LM * len(self.Vocabulary)))
			else:
				return self(current)
		
			"Trigrams"
		else:
			if prev1+" "+prev2+" "+current in self:
				return log10((self.get(prev1+" "+prev2+" "+current, 0) + SMOOTH_LM) / float(self.get(prev1+" "+prev2, 0) + SMOOTH_LM * len(self.Vocabulary)))
			else:
				return self(current, prev2)



	#def setProperty(self, ngram_, n_):
		#ngram = ngram_.split(" ")

		#if ngram_ not in self:
			#if len(ngram) == 1:
				#self.types_uni+=1
			#elif len(ngram) == 2:
				#self.types_bi+=1
			#elif len(ngram) == 3:
				#self.types_tri+=1



	def getVocabulary(self):
		
		return self.Vocabulary
	
	def setVocabulary(self, V):
		
		self.Vocabulary = V

	def increaseN(self, N_):
		self.N += N_



class LanguageModel(dict):

	N_uni = 0
	N_bi = 0
	N_tri = 0
	vocab = {}
	

	def __init__(self):
		arpa = open("LM"+ str(N_GRAM) +".lm","r", encoding="iso-8859-1")
		while arpa.readline().strip() != "\\data\\":
			continue
		
		numberNGrams = [0, 0, 0]
		for i in range(N_GRAM):
			numberNGrams[i] = ( int(arpa.readline().strip().split("=")[1] ) )
		self.N_uni = numberNGrams[0]
		self.N_bi = numberNGrams[1] 
		self.N_tri = numberNGrams[2] 

		n = 0
		while True:
			
			try:
				line = arpa.readline().split()
			except:
				raise ValueError(line)
			
			

			if line:
				if line[0] == "\\end\\":
					break
				elif "-grams:" in line[0]:
					n+=1
				elif n < N_GRAM:
					try:
						self[ " ".join(line[1:n+1])] = (float(line[0]), float(line[n+1]))
					except IndexError:
						self[ " ".join(line[1:n+1]) ] = (float(line[0]), )

				elif n == N_GRAM:
					self[ " ".join(line[1:n+1]) ] = (float(line[0]), )

	@memo2
	def __call__(self, current, prev2=None, prev1=None):
	
		 		
		 		 
		 ### replace unknown tokens with <unk>
		 
		if not current in self.vocab:
			
			if current == "<s>":
				return self["<s>"][0]
			if current == "</s>":
				return self["</s>"][0]			
			
			current = "<unk>"
		s = current
		
		if prev2: 
			if not prev2 in self.vocab:
				prev2 = "<unk>"


			s = prev2 + " " + s
		if prev1: 
			if not prev1 in self.vocab:
				prev1 = "<unk>"


			s = prev1 + " " + s

				
		###  d : (prob, backoff)  ###

		

		if s in self:
			
			unk = log10(1./OOV) if current == "<unk>" else 0
			

			return self[s][0] + unk
		else:
						
			if prev1:
				return self(current, prev2)  + self.backoff(prev2, prev1)
			else:
				return self(current)  + self.backoff(prev2)

	
	def backoff(self, prev2, prev1=None):
		s = prev1 + " " + prev2 if prev1 else prev2
		try:
			return self.get(s)[1]
		except:
			#print("NOT ABLE TO COMPUTE BACKOFF", prev1, prev2)
			return 0
			
	
	def setVocabulary(self,voc):
		self.vocab = voc
	def getVocabulary(self):
		return self.vocab






class ErrorModel(dict):
	unigrams = {}
	bigrams = {}
	numberDeletionAtBeginning = 6413

	def __init__(self):
		pass

	@memo2
	def __call__(self, edit):
		denom = 0
		if edit:
			result = self.get(edit, 0)
			right = re.sub(">", "", edit.split('|')[1])

			if len(right) == 1:
				denom = self.unigrams.get(right, 0)
			elif len(right) == 2:
				denom = self.bigrams.get(right, 0)
			#elif len(right) == 3:
				#denom = self.trigrams.get(right, 0)
			# happens e.g. >e|>
			elif len(right) == 0:
				denom = self.numberDeletionAtBeginning * 100
			else:
				pass#print("'" + right + "' not found in " + edit)

			if denom == 0:
				return 0.000000000000000001
			return (result + SMOOTH_EM) / float(denom + SMOOTH_EM * len(self))
		else:
			return 1



	def setNumberDeletionAtBeginning(self, n):
		self.numberDeletionAtBeginning = n

	def setUnigrams(self, unigrams):
		self.unigrams = unigrams

	def setBigrams(self, bigrams):
		self.bigrams = bigrams

	def getUnigrams(self):
		return self.unigrams

	def getBigrams(self):
		return self.bigrams



def getOriginPath(line):
	path = re.sub(r':', r'', '/nethome/cklaus/Desktop/BachelorThesis/Origin/' + line.split()[0] + '.xml')
	if platform.system() != "Linux":
		path = re.sub(r'/', r'\\', path)
	return path


def getPath(line):
	path = re.sub(r':', r'', '/nethome/cklaus/Desktop/BachelorThesis/corporaTagged/' + line.split()[0] + '.xml.tagged')
	if platform.system() != "Linux":
		path = re.sub(r'/', r'\\', path)
	return path


def getUnigrams(word):
	unigrams = list(filter(lambda x: x != '\n', list(word)))
	return unigrams


def getBigrams(word):
	bigrams = []
	for i in range(len(word) - 1):
		if '\n' not in word[i:i + 2]:
			bigrams.append(word[i:i + 2])
	return bigrams



def buildLanguageModelOLD(alreadyGenerated=False):


	Unigrams = {}
	Bigrams = {}

	
	Vocabulary = {}

	if alreadyGenerated:


		LM = LanguageModelOLD(False)


		for line in open("/nethome/cklaus/Desktop/BachelorThesis/unigrams.count", "r", encoding="iso-8859-1"):
			uniList = line.split('\t')
			Unigrams[uniList[0]] = int(uniList[1])

		for line in open("/nethome/cklaus/Desktop/BachelorThesis/bigrams.count", "r", encoding="iso-8859-1"):
			biList = line.split('\t')
			Bigrams[biList[0]] = int(biList[1])

		#for line in open("/nethome/cklaus/Desktop/BachelorThesis/trigrams.count", "r", encoding="iso-8859-1"):
		#	triList = line.split('\t')
		#	Trigrams[triList[0]] = int(triList[1])
		
		for line in open("/nethome/cklaus/Desktop/BachelorThesis/vocabulary.count", "r", encoding="iso-8859-1"):
			voc = line.split('\t')
			Vocabulary[voc[0]] = int(voc[1])
		
		LM.setVocabulary(Vocabulary)
		


	else:

		LM = LanguageModelOLD(True)


		num_lines = sum(1 for _ in open("/nethome/cklaus/Desktop/BachelorThesis/training_data.txt", "r", encoding="utf8"))

		trainData = open("/nethome/cklaus/Desktop/BachelorThesis/training_data.txt", "r", encoding="utf8")
		count = 0

		for line in trainData:
			path = getPath(line)
			
			previous1, previous2 = None, None

			try:
				tree = ET.ElementTree(file=path)
			except IOError:
				continue

			root = tree.getroot()

			for child in root.getchildren():
				if child.text is not None:
					tokens = [c.split() for c in child.text.split('\n') if c]
					for t in tokens:
						
						# fill character unigrams
						for uni in getUnigrams(t[0]):
							Unigrams[uni] = Unigrams.get(uni, 0) + 1

						# fill character bigrams
						for bi in getBigrams(t[0]):
							Bigrams[bi] = Bigrams.get(bi, 0) + 1

						# fill character trigrams
						#for tri in getTrigrams(t[0]):
						#	Trigrams[tri] = Trigrams.get(tri, 0) + 1



						# fill Language Model ( Word-{Unigrams, Bigrams,Trigrams} )
						#LM.setProperty(t[0], 1)
						LM[t[0]] = LM.get(t[0], 0) + 1
						Vocabulary[t[0]] = Vocabulary.get(t[0], 0) + 1
						LM.increaseN(1)
						
						if N_GRAM > 1 and previous2:
							#LM.setProperty(previous2+" "+t[0], 1)
							LM[previous2+" "+t[0]] = LM.get(previous2+" "+t[0], 0) + 1
						
						if N_GRAM > 2 and previous1 and previous2:
							#LM.setProperty(previous1+" "+previous2+" "+t[0], 1)
							LM[previous1+" "+previous2+" "+t[0]] = LM.get(previous1+" "+previous2+" "+t[0], 0) + 1
						
						previous1 = previous2
						previous2 = t[0]


			count += 1
			sys.stdout.write('\r' + str(count) + " / " + str(num_lines))
			sys.stdout.flush()






		for key in [k for k,v in Vocabulary.items() if v]:
			if Vocabulary[key] < COUNT_THRESHOLD:
				del Vocabulary[key]


		sorted_voc = sorted(Vocabulary, key=Vocabulary.get, reverse=True)
		voc = open("/nethome/cklaus/Desktop/BachelorThesis/vocabulary.count", "w", encoding="iso-8859-1")
		for key in sorted_voc:
			voc.write(key+"	"+str(Vocabulary[key])+"\n")
		voc.close()
		
		LM.setVocabulary(Vocabulary)
		
		
		
		

		print('\n')

		sorted_LM = sorted(LM, key=LM.get, reverse=True)
		sorted_uni = sorted(Unigrams, key=Unigrams.get, reverse=True)
		sorted_bigr = sorted(Bigrams, key=Bigrams.get, reverse=True)
		#sorted_tri = sorted(Trigrams, key=Trigrams.get, reverse=True)

		unigr = open("/nethome/cklaus/Desktop/BachelorThesis/unigrams.count", "w", encoding="iso-8859-1")
		for key in sorted_uni:
			unigr.write(key + "	" + str(Unigrams[key]) + '\n')
		unigr.close()

		bigr = open("/nethome/cklaus/Desktop/BachelorThesis/bigrams.count", "w", encoding="iso-8859-1")
		for key in sorted_bigr:
			bigr.write(key + "	" + str(Bigrams[key]) + '\n')
		bigr.close()

		#tri = open("/nethome/cklaus/Desktop/BachelorThesis/trigrams.count", "w")
		#for key in sorted_tri:
		#	tri.write(key + "	" + str(Trigrams[key]) + '\n')
		#tri.close()


		out = open("/nethome/cklaus/Desktop/BachelorThesis/LanguageModel.count", "w", encoding="iso-8859-1")
		
		for key in sorted_LM:
			out.write(key + "	" + str(LM[key]) + '\n')
		out.close()

	return LM, Unigrams, Bigrams

def buildLanguageModel(alreadyGenerated=False, TestSet = True):

	Unigrams = {}
	Bigrams = {}
	#Trigrams = {}

	Vocab = {}

	if alreadyGenerated:
		
		LM = LanguageModel()

		for line in open("/nethome/cklaus/Desktop/BachelorThesis/unigrams.count", "r", encoding="iso-8859-1"):
			uniList = line.split('\t')
			Unigrams[uniList[0]] = int(uniList[1])

		for line in open("/nethome/cklaus/Desktop/BachelorThesis/bigrams.count", "r", encoding="iso-8859-1"):
			biList = line.split('\t')
			Bigrams[biList[0]] = int(biList[1])

		#for line in open("/nethome/cklaus/Desktop/BachelorThesis/trigrams.count", "r", encoding="iso-8859-1"):
			#triList = line.split('\t')
			#Trigrams[triList[0]] = int(triList[1])
			
		for line in open("/nethome/cklaus/Desktop/BachelorThesis/vocabulary.count", "r", encoding="iso-8859-1"):
			vocList = line.split('\t')
			Vocab[vocList[0]] = int(vocList[1])
			
		LM.setVocabulary(Vocab)

	else:

		num_lines = sum(1 for _ in open("/nethome/cklaus/Desktop/BachelorThesis/training_data.txt", "r", encoding="utf8"))

		trainData = open("/nethome/cklaus/Desktop/BachelorThesis/training_data.txt", "r", encoding="utf8")
		count = 0

		corpus = open("/nethome/cklaus/Desktop/BachelorThesis/corpus.txt", "w", encoding="iso-8859-1")

		for line in trainData:
			linebreaker = 0
			
			if TestSet:
				path = getPath(line)
			else:
				path = '/nethome/cklaus/Desktop/BachelorThesis/CorrectedCorpus/' + line.split()[0][:-1] + ".xml.tagged"

			
			try:	
				tree = ET.ElementTree(file=path)
			except IOError:
				continue

			root = tree.getroot()

			for child in root.getchildren():
				if child.text is not None:
					tokens = [c.split()[0] for c in child.text.split('\n') if c]
					for t in tokens:
						
						# fill character unigrams
						for uni in getUnigrams(t):
							Unigrams[uni] = Unigrams.get(uni, 0) + 1

						# fill character bigrams
						for bi in getBigrams(t):
							Bigrams[bi] = Bigrams.get(bi, 0) + 1

						## fill character trigrams
						#for tri in getTrigrams(t):
							#Trigrams[tri] = Trigrams.get(tri, 0) + 1

						Vocab[t] = Vocab.get(t, 0) + 1

						corpus.write(t + " ")

				linebreaker+=1
				if linebreaker % 50  == 0:
					corpus.write("\n")		



			corpus.write("\n")		


			count += 1
			sys.stdout.write('\r' + str(count) + " / " + str(num_lines))
			sys.stdout.flush()
		corpus.close()
		print('\n')

		global OOV
		OOV = 0

		for key in [k for k,v in Vocab.items() if v]:
			if Vocab[key] < COUNT_THRESHOLD:
				del Vocab[key]
				OOV+=1

		print("OOV",OOV)

		sorted_voc = sorted(Vocab, key=Vocab.get, reverse=True)
		voc = open("/nethome/cklaus/Desktop/BachelorThesis/vocabulary.count", "w")
		for key in sorted_voc:
			voc.write(key+"	"+str(Vocab[key])+"\n")
		voc.close()

		subprocess.call("/nethome/cklaus/tools/srilm/bin/i686-m64/ngram-count   -vocab vocabulary.count   -order "+str(N_GRAM)+"   -no-eos -no-sos    -text corpus.txt  -unk   -write count"+str(N_GRAM)+".count", shell=True)
		 
		
		
		
		print("created COUNT File")
		
				
		smooth = ""
		if N_GRAM != 1:
			for i in range(N_GRAM):
				smooth += " -kndiscount" + str(i+1) + " "
		
		
					
		subprocess.call("/nethome/cklaus/tools/srilm/bin/i686-m64/ngram-count   -vocab vocabulary.count   -order "+str(N_GRAM)+"  -unk -no-eos  -no-sos -read count"+str(N_GRAM)+".count  -lm LM"+str(N_GRAM)+".lm" + smooth, shell=True)
		print("created LANGUAGE MODEL")

		LM = LanguageModel()
		LM.setVocabulary(Vocab)

		sorted_uni = sorted(Unigrams, key=Unigrams.get, reverse=True)
		sorted_bigr = sorted(Bigrams, key=Bigrams.get, reverse=True)
		#sorted_tri = sorted(Trigrams, key=Trigrams.get, reverse=True)
		

		unigr = open("/nethome/cklaus/Desktop/BachelorThesis/unigrams.count", "w")
		for key in sorted_uni:
			unigr.write(key + "	" + str(Unigrams[key]) + '\n')
		unigr.close()

		bigr = open("/nethome/cklaus/Desktop/BachelorThesis/bigrams.count", "w")
		for key in sorted_bigr:
			bigr.write(key + "	" + str(Bigrams[key]) + '\n')
		bigr.close()

		#tri = open("/nethome/cklaus/Desktop/BachelorThesis/trigrams.count", "w")
		#for key in sorted_tri:
			#tri.write(key + "	" + str(Trigrams[key]) + '\n')
		#tri.close()


	return LM , Unigrams, Bigrams

def GroundTruthToTxt():
	
	num_lines = sum(1 for _ in open("/nethome/cklaus/Desktop/BachelorThesis/test_data.txt", "r", encoding="utf8"))

	testData = open("/nethome/cklaus/Desktop/BachelorThesis/test_data.txt", "r", encoding="utf8")
	count = 0

	corpus = open("/nethome/cklaus/Desktop/BachelorThesis/groundTruth.txt", "w")

	for file in testData:
		linebreaker = 0

		if platform.system() == "Linux":
			file = file.split()[0][:-1] + ".xml.tagged"
			pathGT = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/GroundTruth/' + file
		else:
			file = re.sub(r'/', r'\\\\', file.split()[0][:-1] + ".xml.tagged")
			pathGT = r'testSet\\GroundTruth\\' + file
		
		try:
			with open(pathGT, 'r', encoding="iso-8859-1") as f:
				tree = ET.parse(f)		
		except IOError:
			print("File not found!")
			continue

		root = tree.getroot()

		for child in root.getchildren():
			linebreaker+=1

			if child.text is not None:
				tokens = [c.split()[0] for c in child.text.split('\n') if c]
				for t in tokens:
					corpus.write(t+" ")
			if linebreaker % 50 == 0:
				corpus.write("\n")		

		corpus.write("\n")		

		count += 1
		sys.stdout.write('\r' + str(count) + " / " + str(num_lines))
		sys.stdout.flush()
	corpus.close()


def readRules():
	
	######
	return
	######
	
	file = codecs.open("/nethome/cklaus/Desktop/BachelorThesis/cleanDifferences.count", "r", encoding="utf8")
	rules = {}

	for line in file:

		lineList = line.split()
		try:
			lineList[4]
			rules[lineList[1]] = (lineList[3] + " " + lineList[4], lineList[0])
		except IndexError:
			rules[lineList[1]] = (lineList[3], lineList[0])

	return rules


def buildErrorModel(rules, Unigrams, Bigrams, alreadyGenerated=False):
	edits = ErrorModel()
	edits.setUnigrams(Unigrams)
	edits.setBigrams(Bigrams)

	#if alreadyGenerated:
	editFile = open("edits.count", "r", encoding="iso-8859-1")
	for line in editFile:
		lineList = line.split('\t')
		edits[lineList[0]] = int(lineList[1])

	## solve problem with deletion at beginning
	summe = 0
	for key in edits:
		spl = key.split('|')[1]
		if spl == ">":
			summe += edits[key]
	edits.setNumberDeletionAtBeginning(summe)

	#~ else:
		#~ count = 0
		#~ num_lines = sum(1 for _ in open("/nethome/cklaus/Desktop/BachelorThesis/training_data.txt", "r", encoding="utf8"))
		#~ allTitles = open("/nethome/cklaus/Desktop/BachelorThesis/training_data.txt", "r", encoding="utf8")
		#~ rules_cpy = {k: rules[k] for k in rules}

		#~ for line in allTitles:

			#~ path = getOriginPath(line)

			#~ try:
				#~ tree = ET.ElementTree(file=path)
			#~ except IOError:
				#~ print("FILE NOT FOUND !!")
				#~ print(path)
				#~ continue

			#~ root = tree.getroot()

			#~ for child in root.getchildren():

				#~ if child.text is not None:
					#~ childWordList = child.text.split()

					#~ for word in childWordList:

						#~ # WRONG WRITTEN WORD
						#~ if word in rules:

							#~ ### monitor which rules were already applied
							#~ try:
								#~ del rules_cpy[word]
							#~ except KeyError:
								#~ pass
							
							#~ if " " in rules[word][0]:
								#~ continue

							#~ for edit in alignWords(word, rules[word][0]):
								#~ edits[edit] = edits.get(edit, 0) + 1

						#~ # CORRECT WRITTEN WORD
						#~ else:
							#~ for c in list(word):
								#~ edits[c + "|" + c] = edits.get(c + "|" + c, 0) + 1
							#~ for c in getBigrams(word):
								#~ edits[c + "|" + c] = edits.get(c + "|" + c, 0) + 1
							#~ #for c in getTrigrams(word):
							#~ #	edits[c + "|" + c] = edits.get(c + "|" + c, 0) + 1

								
								

			#~ count += 1
			#~ sys.stdout.write('\r' + str(count) + " / " + str(num_lines))
			#~ sys.stdout.flush()
		#~ print('\n')

		#~ ### fidding around with unused rules
		#~ for w in rules_cpy:
			#~ for edit in [e for e in alignWords(w, rules_cpy[w][0])]:
				#~ edits[edit] = edits.get(edit, 0) + int(rules_cpy[w][1])

		#~ 'MISSING UNIGRAMS / BIGRAMS'

		#~ dic = {}
		#~ for edit in edits:
			#~ spl = edit.split("|")[1]
			#~ if spl in edits.getUnigrams() or spl in edits.getBigrams():
				#~ pass
			#~ else:
				#~ dic[spl] = dic.get(spl, 0) + edits[edit]
		#~ uni = open("/nethome/cklaus/Desktop/BachelorThesis/unigrams.count", "a", encoding="iso-8859-1")
		#~ bi = open("/nethome/cklaus/Desktop/BachelorThesis/bigrams.count", "a", encoding="iso-8859-1")
		#~ #tri = open("/nethome/cklaus/Desktop/BachelorThesis/trigrams.count", "a")
		#~ for d in dic:
			#~ if len(d) == 1:
				#~ uni.write(d + "\t" + str(dic[d]) + "\n")
				
				#~ tmp = edits.getUnigrams()
				#~ tmp.update({d: dic[d]})
				#~ edits.setUnigrams(tmp)
			#~ elif len(d) == 2:
				#~ bi.write(d + "\t" + str(dic[d]) + "\n")

				#~ tmp = edits.getBigrams()
				#~ tmp.update({d: dic[d]})
				#~ edits.setBigrams(tmp)
			#~ #else:
				#~ #tri.write(d + "\t" + str(dic[d]) + "\n")

				#~ #tmp = edits.getTrigrams()
				#~ #tmp.update({d: dic[d]})
				#~ #edits.setTrigrams(tmp)

		#~ uni.close()
		#~ bi.close()
		#~ #tri.close()

		#~ ## solve problem with deletion at beginning
		#~ summe = 0
		#~ for key in edits:
			#~ spl = key.split('|')[1]
			#~ if spl == ">":
				#~ summe += edits[key]
		#~ edits.setNumberDeletionAtBeginning(summe)

		#~ ## WRITE TO FILE
		#~ sorted_EM = sorted(edits, key=edits.get, reverse=True)
		#~ out = open("/nethome/cklaus/Desktop/BachelorThesis/edits.count", "w", encoding="iso-8859-1")

		#~ for key in sorted_EM:
			#~ out.write(key + "	" + str(edits[key]) + '\n')
		#~ out.close()

	return edits


def buildBlackList():
	
	########
	return
	#######
	
	global blackList
	file = codecs.open("/nethome/cklaus/Desktop/BachelorThesis/cleanDifferences.count", "r", encoding="utf8")
	blackList = {}

	for line in file:
		lineList = line.split()

		try:
			lineList[4]
			blackList[lineList[1]] = lineList[3] + " " + lineList[4]
		except IndexError:
			#continue
			blackList[lineList[1]] = lineList[3]

	file.close()
	return blackList


def alignWordsOLD(a, b):
	# https://github.com/alevchuk/pairwise-alignment-in-python

	def zeros(shape):
		retval = []
		for x in range(shape[0]):
			retval.append([])
			for y in range(shape[1]):
				retval[-1].append(0)
		return retval

	match_award = 10
	mismatch_penalty = -5
	gap_penalty = -5  # both for opening and extanding

	def match_score(alpha, beta):
		if alpha == beta:
			return match_award
		elif alpha == '°' or beta == '°':
			return gap_penalty
		else:
			return mismatch_penalty

	def finalize(aligA, aligB):


		edits = []

		aligA = aligA[::-1]  # reverse sequence 1
		aligB = aligB[::-1]  # reverse sequence 2

		temp = ">"

		for i in range(max(len(aligA), len(aligB))):

			if aligA[i] == aligB[i]:
				edits.append(aligA[i] + '|' + aligA[i])  # eq

				temp += aligA[i]

			elif aligA[i] == '°':
				if i == 0:
					edits.append('>|>' + aligB[i])  # insert

				else:
					edits.append(temp[-1] + '|' + temp[-1] + aligB[i])  # insert

				temp += aligB[i]

			elif aligB[i] == '°':
				if i == 0:
					edits.append('>' + aligA[i] + '|>')  # delete
				else:
					edits.append(temp[-1] + aligA[i] + '|' + temp[-1])  # delete

			elif aligA[i] != aligB[i]:
				edits.append(aligA[i] + '|' + aligB[i])  # repl
				temp += aligB[i]

		return edits

	def needle(seq1, seq2):
		m, n = len(seq1), len(seq2)  # length of two sequences

		# Generate DP table and traceback path pointer matrix
		score = zeros((m + 1, n + 1))  # the DP table

		# Calculate DP table
		for i in range(0, m + 1):
			score[i][0] = gap_penalty * i
		for j in range(0, n + 1):
			score[0][j] = gap_penalty * j
		for i in range(1, m + 1):
			for j in range(1, n + 1):
				match = score[i - 1][j - 1] + match_score(seq1[i - 1], seq2[j - 1])
				delete = score[i - 1][j] + gap_penalty
				insert = score[i][j - 1] + gap_penalty
				score[i][j] = max(match, delete, insert)

		# Traceback and compute the alignment
		aligA, aligB = '', ''
		i, j = m, n  # start from the bottom right cell

		while i > 0 and j > 0:  # end toching the top or the left edge
			score_current = score[i][j]
			score_diagonal = score[i - 1][j - 1]
			score_up = score[i][j - 1]
			score_left = score[i - 1][j]

			if score_current == score_left + gap_penalty:
				aligA += seq1[i - 1]
				aligB += '°'
				i -= 1
			elif score_current == score_up + gap_penalty:
				aligA += '°'
				aligB += seq2[j - 1]
				j -= 1
			elif score_current == score_diagonal + match_score(seq1[i - 1], seq2[j - 1]):  # and seq1[i-1] == seq2[j-1] :
				aligA += seq1[i - 1]
				aligB += seq2[j - 1]
				i -= 1
				j -= 1

			# Finish tracing up to the top left cell
		while i > 0:
			aligA += seq1[i - 1]
			aligB += '°'
			i -= 1
		while j > 0:
			aligA += '°'
			aligB += seq2[j - 1]
			j -= 1

		return finalize(aligA, aligB)

	return needle(a, b)


def alignWords(a, b):
	# https://github.com/alevchuk/pairwise-alignment-in-python

	def zeros(shape):
		retval = []
		for x in range(shape[0]):
			retval.append([])
			for y in range(shape[1]):
				retval[-1].append(0)
		return retval

	match_award = 10
	mismatch_penalty = -5
	gap_penalty = -5  # both for opening and extanding

	def match_score(alpha, beta):
		if alpha == beta:
			return match_award
		elif alpha == '°' or beta == '°':
			return gap_penalty
		else:
			return mismatch_penalty

	def finalize(aligA, aligB):

		edits = []

		aligA = aligA[::-1]  # reverse sequence 1
		aligB = aligB[::-1]  # reverse sequence 2

		# temp = ">"

		first = ""
		second = ""

		for i in range(max(len(aligA), len(aligB))):

			if aligA[i] == aligB[i]:

				if first or second:
					edits.append(first + '|' + second)
				# if second:

				first = ""
				second = ""

				edits.append(aligA[i] + '|' + aligA[i])  # eq

			# temp+=aligA[i]



			elif aligA[i] == '°':
				if i == 0:
					first = '>'
					second = '>' + aligB[i]
				# edits.append('>|>'+aligB[i]) #insert

				else:
					# edits.append(temp[-1]+'|'+temp[-1]+aligB[i]) #insert
					if first:
						second += aligB[i]
					else:
						first += aligA[i - 1]
						second += aligB[i - 1] + aligB[i]

					# temp+=aligB[i]

			elif aligB[i] == '°':
				if i == 0:
					first = '>' + aligA[i]
					second = '>'
				# edits.append('>'+aligA[i]+'|>')   #delete
				else:
					# edits.append(temp[-1]+aligA[i]+'|'+temp[-1]) #delete
					if second:
						first += aligA[i]
					else:
						first += aligA[i - 1] + aligA[i]
						second += aligB[i - 1]

					# first+=aligA[i]

			elif aligA[i] != aligB[i]:
				first += aligA[i]
				second += aligB[i]

			# edits.append(aligA[i]+'|'+aligB[i]) #repl
			# temp+=aligB[i]

		if first or second:
			edits.append(first + '|' + second)

		return edits

	def needle(seq1, seq2):
		m, n = len(seq1), len(seq2)  # length of two sequences

		# Generate DP table and traceback path pointer matrix
		score = zeros((m + 1, n + 1))  # the DP table

		# Calculate DP table
		for i in range(0, m + 1):
			score[i][0] = gap_penalty * i
		for j in range(0, n + 1):
			score[0][j] = gap_penalty * j
		for i in range(1, m + 1):
			for j in range(1, n + 1):
				match = score[i - 1][j - 1] + match_score(seq1[i - 1], seq2[j - 1])
				delete = score[i - 1][j] + gap_penalty
				insert = score[i][j - 1] + gap_penalty
				score[i][j] = max(match, delete, insert)

		# Traceback and compute the alignment
		aligA, aligB = '', ''
		i, j = m, n  # start from the bottom right cell

		while i > 0 and j > 0:  # end toching the top or the left edge
			score_current = score[i][j]
			score_diagonal = score[i - 1][j - 1]
			score_up = score[i][j - 1]
			score_left = score[i - 1][j]

			if score_current == score_left + gap_penalty:
				aligA += seq1[i - 1]
				aligB += '°'
				i -= 1
			elif score_current == score_up + gap_penalty:
				aligA += '°'
				aligB += seq2[j - 1]
				j -= 1
			elif score_current == score_diagonal + match_score(seq1[i - 1], seq2[j - 1]):  # and seq1[i-1] == seq2[j-1] :
				aligA += seq1[i - 1]
				aligB += seq2[j - 1]
				i -= 1
				j -= 1

			# Finish tracing up to the top left cell
		while i > 0:
			aligA += seq1[i - 1]
			aligB += '°'
			i -= 1
		while j > 0:
			aligA += '°'
			aligB += seq2[j - 1]
			j -= 1

		return finalize(aligA, aligB)

	return needle(a, b)


@memo2
def editProbability(EM, word, candidate):
	return sum(log10(EM(edit)) for edit in alignWords(word, candidate))


@memo2
def edProb(EM, edits):
	return sum(log10(EM(e)) for e in edits.split('<>'))


def assignPredecessors(prev2, current):
	global N_GRAM

	if N_GRAM == 2:
		return None, current
	elif N_GRAM == 3:
		return prev2, current
	else:
		return None, None



def correct(Vocabulary, LM, EM, text, previous2=None, previous1=None):
	global blackList
	tag_blackList = ["SENT", ",", ":", "``", "''", "(", ")", "SYM"]
	tokens = [c.split() for c in text.split('\n') if c]

	i = -1
	for t in tokens:
		i += 1

		## some tags do not need a correction, like interpunctions
		if t[1] in tag_blackList:
			previous1, previous2 = assignPredecessors(previous2, t[0])
			continue

		if t[0] in blackList:

			### Rule Based Space Insertion
			if " " in blackList[t[0]]:
				first, second = blackList[t[0]].split(" ")[0], blackList[t[0]].split(" ")[1]
				t[0], t[1], t[2] = first, t[1], t[2]
				tokens.insert(i + 1, [second, t[1], t[2]])
			### Rule Based Word Replacement
			else:
				t[0], t[1], t[2] = blackList[t[0]], t[1], t[2]

			
			#####################################################################
			#previous1, previous2 = assignPredecessors(previous2, t[0])			#
			#continue															#
			#####################################################################


		prefix = ""
		suffix = ""
		if t[0][0] in ["*", "'", "\"", "."] and len(t[0]) > 1:
			prefix = t[0][0]
			t[0] = t[0][1:]
		if t[0]:
			if t[0][-1] in ["*", "'", "\"", "."] and len(t[0]) > 1:
				suffix = t[0][-1]
				t[0] = t[0][:-1]

		maxi = -np.inf
		for c in edits(Vocabulary, EM, t[0]):
			
			res = LM(c, previous2, previous1) * PRIOR_WEIGHT + editProbability(EM, t[0], c)			
			if res > maxi:
				maxi = res
				correctedWord = c
		
		previous1, previous2 = assignPredecessors(previous2, t[0])

		t[0], t[1], t[2] = prefix + correctedWord + suffix, t[1], t[2]

	return "\n" + '\n'.join(['\t'.join(t) for t in tokens]) + "\n", previous1, previous2





def correctPlainText(LM, EM, tokens, previous2=None, previous1=None):
	
	Vocabulary = LM.getVocabulary()
	
	if not isinstance(tokens, (list,)):
		tokens = [c for c in tokens.split()]

	for i, t in enumerate(tokens):
		
		
		prefix = ""
		suffix = ""
		if t[0] in string.punctuation and len(t) > 1:
			prefix = t[0]
			t = t[1:]
		if t:
			if t[-1] in string.punctuation and len(t) > 1:
				suffix = t[-1]
				t = t[:-1]
				

		maxi = -np.inf
		for c in edits(Vocabulary, EM, t):
			
			res = LM(c, previous2, previous1) * PRIOR_WEIGHT + editProbability(EM, t, c)			
			if res > maxi:
				maxi = res
				correctedWord = c

			
		previous1, previous2 = assignPredecessors(previous2, t)

		t = prefix + correctedWord + suffix
		tokens[i] = t

	return ' '.join(t for t in tokens)












@memo3
def edits(Vocabulary, EM, word, d=DISTANCE_LIMIT):

	alphabet = "ACBEDGFIHKJMLONQPSRUTWVYXZa`cbedgfihkjmlonqpsrutwvyxz'-"

	# Return a dict of {correct: edit} pairs within d edits of word.
	results = {}

	def editsR(hd, tl, d, edits):

		def ed(L, R):
			return edits + [R + '|' + L]

		C = hd + tl
		if C in Vocabulary:
			e = '<>'.join(edits)

			if C not in results:
				results[C] = e
			else:
				results[C] = max(results[C], e, key=lambda e: edProb(EM, e))

		if d <= 0:
			return
		extensions = [hd + c for c in alphabet if hd + c in PREFIXES]
		p = (hd[-1] if hd else '>')  ## previous character
		## Insertion
		for h in extensions:
			editsR(h, tl, d - 1, ed(p + h[-1], p))
		if not tl:
			return
		## Deletion
		editsR(hd, tl[1:], d - 1, ed(p, p + tl[0]))
		for h in extensions:
			if h[-1] == tl[0]:  ## Match
				editsR(h, tl[1:], d, edits)
			else:  ## Replacement
				editsR(h, tl[1:], d - 1, ed(h[-1], tl[0]))
			## Body of edits:

	editsR('', word, d, [])

	results[word] = ''
	return results



def TextAlignment(textA, textB):
	def zeros(shape):
		retval = []
		for x in range(shape[0]):
			retval.append([])
			for y in range(shape[1]):
				retval[-1].append(0)
		return retval

	match_award = 10
	mismatch_penalty = -5
	gap_penalty = -5  # both for opening and extanding

	def match_score(alpha, beta):
		if alpha == beta:
			return match_award
		elif alpha == '°' or beta == '°':
			return gap_penalty
		else:
			return mismatch_penalty

	def needle(seq1, seq2):
		m, n = len(seq1), len(seq2)  # length of two sequences

		# Generate DP table and traceback path pointer matrix
		score = zeros((m + 1, n + 1))  # the DP table

		seq1 = seq1[::-1]  # reverse sequence 1
		seq2 = seq2[::-1]  # reverse sequence 2

		# Calculate DP table
		for i in range(0, m + 1):
			score[i][0] = gap_penalty * i
		for j in range(0, n + 1):
			score[0][j] = gap_penalty * j
		for i in range(1, m + 1):
			for j in range(1, n + 1):
				match = score[i - 1][j - 1] + match_score(seq1[i - 1], seq2[j - 1])
				delete = score[i - 1][j] + gap_penalty
				insert = score[i][j - 1] + gap_penalty
				score[i][j] = max(match, delete, insert)

		# Traceback and compute the alignment
		aligA, aligB = '', ''
		i, j = m, n  # start from the bottom right cell

		while i > 0 and j > 0:  # end toching the top or the left edge
			score_current = score[i][j]
			score_diagonal = score[i - 1][j - 1]
			score_up = score[i][j - 1]
			score_left = score[i - 1][j]

			if score_current == score_diagonal + match_score(seq1[i - 1], seq2[j - 1]):  # and seq1[i-1] == seq2[j-1] :
				aligA += seq1[i - 1] + " "
				aligB += seq2[j - 1] + " "
				i -= 1
				j -= 1
			elif score_current == score_up + gap_penalty:
				aligA += '° '
				aligB += seq2[j - 1] + " "
				j -= 1
			elif score_current == score_left + gap_penalty:
				aligA += seq1[i - 1] + " "
				aligB += '° '
				i -= 1

		# Finish tracing up to the top left cell
		while i > 0:
			aligA += seq1[i - 1] + " "
			aligB += '° '
			i -= 1
		while j > 0:
			aligA += '° '
			aligB += seq2[j - 1] + " "
			j -= 1

		return aligA.split(), aligB.split()

	return needle(textA, textB)


def OverlapAnalysis(textA, textB):
	## computes the percentage of changes in the first text

	textA = [c.split()[0] for c in textA.split('\n') if c]
	textB = [c.split()[0] for c in textB.split('\n') if c]

	numberwords = len(textA)
	changes = 0

	textA, textB = TextAlignment(textA, textB)

	for i in range(len(textA)):

		if textA[i] == "°":  ## word split, counts as one change
			continue
		elif textA[i] != textB[i]:  ## difference
			changes += 1

	return changes, numberwords


def metrics(A_, B_, text_):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	splitted = False

	A_ = [c.split()[0] for c in A_.split('\n') if c]
	B_ = [c.split()[0] for c in B_.split('\n') if c]
	text_ = [c.split()[0] for c in text_.split('\n') if c]

	## Multiple Alignment ##
	A, B = TextAlignment(A_, B_)
	text, B = TextAlignment(text_, B)
	text, A = TextAlignment(text, A)

	## bringing the alignments to equal length ##
	if len(A) != len(B) or len(B) != len(text) or len(text) != len(A):
		A, B = TextAlignment(A_, B_)
		text, B = TextAlignment(text_, B)
		text, A = TextAlignment(text, A)
	if len(A) != len(B) or len(B) != len(text) or len(text) != len(A):
		text, A = TextAlignment(text_, A_)
		A, B = TextAlignment(A, B_)
		text, B = TextAlignment(text, B)

	if len(A) != len(B) or len(B) != len(text) or len(text) != len(A):
		text, B = TextAlignment(text_, B_)
		A, B = TextAlignment(A_, B)
		text, A = TextAlignment(text, A)

	if len(A) != len(B) or len(B) != len(text) or len(text) != len(A):
		text, B = TextAlignment(text_, B_)
		A, text = TextAlignment(A_, text)
		B, A = TextAlignment(B, A)

	if len(A) != len(B) or len(B) != len(text) or len(text) != len(A):
		print(A, len(A))
		print(B, len(B))
		print(text, len(text))
		raise ValueError("Alignments don't have the same length")

	if text.count("°") > 20 or A.count("°") > 20 or B.count("°") > 20:
		print(len(A))
		print(len(B))
		print(len(text))
		print(A)
		print(B)
		print(text)
		raise ValueError("TOO MANY ° (Stars)! I think there is a mistakes with the <normalised...> shit")

	## Calculate metric components (TP, ...)
	for i in range(len(text)):

		if (not splitted) and (A[i] == "°" or B[i] == "°" or text[i] == "°"):
			splitted = True
			continue
		else:
			splitted = False

		if A[i] != B[i]:
			if B[i] == text[i]:
				TP += 1
			else:
				FN += 1

		else:
			if A[i] == text[i]:
				TN += 1
			else:
				FP += 1

	return TP, TN, FP, FN



def correctDocument(Vocabulary, LM, EM, fileName, TestSet = True):
	
	previous1, previous2 = None, None
	count = 0
	
	if TestSet:
		test = "testSet/"
	else: 
		test  = ""

	if platform.system() == "Linux":
		path = '/nethome/cklaus/Desktop/BachelorThesis/'+test+'Origin/' + fileName.split()[0][:-1] + ".xml.tagged"
	else:
		path = re.sub(r'/', r'\\\\', r'testSet\\Origin\\' + fileName.split()[0][:-1] + ".xml.tagged")

	try:
		tree = ET.ElementTree(file=path)
	except IOError:
		print("FILE NOT FOUND !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " + fileName)
		return

	root = tree.getroot()
	
	for child in root.getchildren():

		count += 1

		if child.text is not None:
			child.text, previous1, previous2 = correct(Vocabulary, LM, EM, child.text, previous2, previous1)


		## quick and dirty method to reset cache if RAM is too full
		if False:#count % 30 == 0:
			LM()
			EM()
			editProbability()
			edProb()
			edits()

	## quick and dirty method to reset cache after every file
	if True:
		LM()
		EM()
		editProbability()
		edProb()
		edits()

	if TestSet:
		tree.write(re.sub(r'Origin', r'NoisyChannel', path))
	else:
		tree.write(re.sub(r'Origin', r'CorrectedCorpus', path))

	sys.stdout.write("Done!	"+fileName)
	sys.stdout.flush()
		

def correctionTestSet(LM, EM):


	testFiles = open("/nethome/cklaus/Desktop/BachelorThesis/test_data.txt")
	Vocabulary = LM.getVocabulary()



	num_cores = multiprocessing.cpu_count()
	
	num_cores -= 2
	

	Parallel(n_jobs=num_cores)(delayed(correctDocument)(Vocabulary, LM, EM, name) for name in testFiles)	
	


def testIt(file, algorithm="<noisy>"):
	TP, TN, FP, FN = 0,0,0,0

	if platform.system() == "Linux":
		file = file.split()[0][:-1] + ".xml.tagged"
		pathOri = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/Origin/' + file
		pathGT = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/GroundTruth/' + file

		if algorithm == "<rules>":
			prefix = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/Rules/'
		elif algorithm == "<noisy>":
			prefix = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/NoisyChannel/'
		elif algorithm == "<norvigUni>":
			prefix = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/NorvigUnigrams/'
		elif algorithm == "<norvigBi>":
			prefix = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/NorvigBigrams/'
		else:
			raise ValueError("algorithm option not known!")
		path = prefix + file
	else:
		file = re.sub(r'/', r'\\\\', file.split()[0][:-1] + ".xml.tagged")
		pathOri = r'testSet\\Origin\\' + file
		pathGT = r'testSet\\GroundTruth\\' + file

		if algorithm == "<rules>":
			prefix = r'testSet\\Rules\\'
		elif algorithm == "<noisy>":
			prefix = r'testSet\\NoisyChannel\\'
		elif algorithm == "<norvigUni>":
			prefix = r'testSet\\NorvigUnigrams\\'
		elif algorithm == "<norvigBi>":
			prefix = r'testSet\\NorvigBigrams\\'
		else:
			raise ValueError("algorithm option not known!")
		path = prefix + file

	try:
		treeOri = ET.ElementTree(file=pathOri)
		with open(pathGT, 'r', encoding="iso-8859-1") as f:
			treeGT = ET.parse(f)
		tree = ET.ElementTree(file=path)
	except IOError:
		print("FILE NOT FOUND !! " + file)

	rootOri = treeOri.getroot()
	rootGT = treeGT.getroot()
	root = tree.getroot()

	childsOri = [child.text for child in rootOri.getchildren()]
	childsGT = [child.text for child in rootGT.getchildren()]
	childs = [child.text for child in root.getchildren()]

	if not (len(childsOri) == len(childsGT) and len(childsGT) == len(childs)):
		print("INCONSISTENT NUMBER OF CHILDS")


	for i in range(len(childs)):

		if childsOri[i] is not None and childsGT[i] is not None and childs[i] is not None:
			tmp_TP, tmp_TN, tmp_FP, tmp_FN = metrics(childsOri[i], childsGT[i], childs[i])
			TP+=tmp_TP
			TN+=tmp_TN
			FP+=tmp_FP
			FN+=tmp_FN

		else:
			raise Exception("There is a inconsistent amount of pages")


	return (TP, TN, FP, FN, file) 


def testMetrics(algorithm="<noisy>"):
	TP, TN, FP, FN = 1, 1, 1, 1

	testFiles = open("/nethome/cklaus/Desktop/BachelorThesis/test_data.txt")

	

	data = [ (name, algorithm)  for name in [file for file in testFiles]] 
	num_cores = multiprocessing.cpu_count()

	p = multiprocessing.Pool(num_cores)
	
	for v in [ (i[0],i[1],i[2],i[3],i[4]) for i in p.starmap(testIt, data)]:
		
		TP+=v[0]
		TN+=v[1]
		FP+=v[2]
		FN+=v[3]
		
		
		

	precision = TP / float(TP + FP)
	recall = TP / float(TP + FN)
	f_score = (2 * precision * recall) / float(precision + recall)


	#sys.stdout.write("COMPONENTS: "+str(TP)+" "+ str(TN)+" "+ str(FP) +" "+ str(FN)+"\n")
	#sys.stdout.flush()
	sys.stdout.write(str(precision)+"\t"+str(recall)+"\t"+str(f_score)+"\n")
	sys.stdout.flush()	
	


	return precision, recall, f_score

def computePerplexity():

	subprocess.call("/nethome/cklaus/tools/srilm/bin/i686-m64/ngram -lm LM" + str(N_GRAM) + ".lm -unk -ppl groundTruth.txt -order "+str(N_GRAM), shell=True)

def testChangingPercentage(algorithm="<noisy>"):
	changes, numberwords = 0, 0
	testFiles = open("/nethome/cklaus/Desktop/BachelorThesis/test_data.txt")

	outerCount = 0
	for file in testFiles:
		outerCount += 1

		count = 0

		if platform.system() == "Linux":
			file = file.split()[0][:-1] + ".xml.tagged"
			pathOri = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/Origin/' + file

			if algorithm == "<rules>":
				prefix = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/Rules/'
			elif algorithm == "<noisy>":
				prefix = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/NoisyChannel/'
			elif algorithm == "<norvigUni>":
				prefix = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/NorvigUnigrams/'
			elif algorithm == "<norvigBi>":
				prefix = r'/nethome/cklaus/Desktop/BachelorThesis/testSet/NorvigBigrams/'
			else:
				raise ValueError("algorithm option not known!")
			path = prefix + file
		else:
			file = re.sub(r'/', r'\\\\', file.split()[0][:-1] + ".xml.tagged")
			pathOri = r'testSet\\Origin\\' + file

			if algorithm == "<rules>":
				prefix = r'testSet\\Rules\\'
			elif algorithm == "<noisy>":
				prefix = r'testSet\\NoisyChannel\\'
			elif algorithm == "<norvigUni>":
				prefix = r'testSet\\NorvigUnigrams\\'
			elif algorithm == "<norvigBi>":
				prefix = r'testSet\\NorvigBigrams\\'
			else:
				raise ValueError("algorithm option not known!")
			path = prefix + file

		try:
			treeOri = ET.ElementTree(file=pathOri)
			tree = ET.ElementTree(file=path)
		except IOError:
			print("FILE NOT FOUND in changingPercentage!! " + file)
			continue

		rootOri = treeOri.getroot()
		root = tree.getroot()

		childsOri = [child.text for child in rootOri.getchildren()]
		childs = [child.text for child in root.getchildren()]

		numPages = sum(1 for _ in rootOri.getchildren())

		if not (len(childsOri) == len(childs)):
			print("INCONSISTENT NUMBER OF CHILDS")

		for i in range(len(childs)):
			count += 1

			if childsOri[i] is not None and childs[i] is not None:
				tmpchanges, tmpnumberwords = OverlapAnalysis(childsOri[i], childs[i])
				changes += tmpchanges
				numberwords += tmpnumberwords
			else:
				raise Exception("There is a inconsistent amount of pages")

			sys.stdout.write('\r' + str(count) + " / " + str(numPages))
			sys.stdout.flush()


	print("\r" + "Changing percentage: {0}".format(changes / float(numberwords)))






def computePerplexityOLD(LM):

	for ngram in range(1, N_GRAM+1):

		if ngram == 1:
			print("Perplexity Unigram")
		elif ngram == 2:
			print("Perplexity Bigram")
		else:
			print("Perplexity Trigram")


		base = 2
		NumberWords = 0
		logProb = 0

		testFiles = open("/nethome/cklaus/Desktop/BachelorThesis/test_data.txt")
		for file in testFiles:

			if platform.system() == "Linux":
				path = '/nethome/cklaus/Desktop/BachelorThesis/testSet/GroundTruth/' + file.split()[0][:-1] + ".xml.tagged"
			else:
				path = 'testSet\\GroundTruth\\' + re.sub(r'/', r'\\\\', file.split()[0][:-1] + ".xml.tagged")

			previous1, previous2 = None, None

			try:
				with open(path, 'r', encoding="iso-8859-1") as f:
					tree = ET.parse(f)
			except IOError:
				print("FILE NOT FOUND !! " + file)
				continue

			root = tree.getroot()
			childs = [child.text for child in root.getchildren()]

			for i in range(len(childs)):

				if childs[i] is not None:
					tmp_NW, tmp_LOGPROB, tmp_previous1, tmp_previous2 = perplexityOLD(childs[i], LM, base, ngram, previous2, previous1)
					NumberWords += tmp_NW
					logProb += tmp_LOGPROB
					previous1 = tmp_previous1
					previous2 = tmp_previous2


		perplexityValue = base ** ((-1. / NumberWords) * logProb)

		print("Perplexity:", perplexityValue)


def perplexityOLD(text, LM, base, ngram, previous2=None, previous1=None):
	sumWords = 0
	logProb = 0

	tokens = [c.split()[0] for c in text.split('\n') if c]
	for t in tokens:
	
		sumWords += 1
		logProb += log(10**LM(t, previous2, previous1), base)



		if ngram == 2:
			previous2 = t
		elif ngram == 3:
			previous1 = previous2
			previous2 = t

	return sumWords, logProb, previous1, previous2



def createHistograms(LM, EM):


	def LanguageModelNoisyHistogram(LM):
		num_lines = sum(1 for _ in open("test_data.txt", "r", encoding="utf8"))
		testFiles = open("test_data.txt")


		count = 0
		if platform.system() == "Linux":
			prefix = r'testSet/NoisyChannel/'
		else:
			prefix = r'testSet\\NoisyChannel\\'

		logProbs = []
		sumWords = 0

		if platform.system() == "Linux":
			out = open("histograms/LM_Noisy.dat", "w")
		else:
			out = open("histograms\\LM_Noisy.dat", "w")

		for file in testFiles:
			count += 1

			file = file.split()[0][:-1] + ".xml.tagged"
			path = prefix + file

			try:
				with open(path, 'r', encoding="iso-8859-1") as f:
					tree = ET.parse(f)
			except IOError:
				print("FILE NOT FOUND !! " + file)
				continue

			previous1 = None
			previous2 = None

			root = tree.getroot()
			childs = [child.text for child in root.getchildren()]

			for i in range(len(childs)):

				if childs[i] is not None:

					tokens = [c.split()[0] for c in childs[i].split('\n') if c]
					for t in tokens:
						
						sumWords+=1
						logProbs.append(LM(t, previous2,previous1))
						out.write(str(-LM(t, previous2,previous1)) + '\n')

						previous1 = previous2
						previous2 = t

			sys.stdout.write('\r' + str(count) + " / " + str(num_lines))
			sys.stdout.flush()
		print("\n")

		testFiles.close()
		out.close()



		mean  = sum( -key/len(logProbs)  for key in logProbs)

		variance = sum(  ((-key - mean) ** 2) / len(logProbs)    for key in logProbs)

		standardDeviation = sqrt(variance)

		print("NOISY LM")
		print("Mean:", mean)
		print("Variance:", variance)
		print("StandardDeviation:", standardDeviation)

	def ErrorModelNoisyHistogram(edits):


		count = 0
		num_lines = sum(1 for _ in open("test_data.txt", "r", encoding="utf8"))
		testFiles = open("test_data.txt", "r", encoding="utf8")


		logProbs = []
		sumEdits = 0


		if platform.system() == "Linux":
			out = open("histograms/EM_Noisy.dat", "w")
		else:
			out = open("histograms\\EM_Noisy.dat", "w")


		x = []
		y = []


		for line in testFiles:

			count += 1

			if platform.system() == "Linux":
				path = r'testSet/NoisyChannel/' + line.split()[0][:-1] + ".xml.tagged"
				pathOri = r'testSet/Origin/' + line.split()[0][:-1] + ".xml.tagged"
			else:
				path = r'testSet\\NoisyChannel\\' + re.sub(r'/', r'\\\\', line.split()[0][:-1] + ".xml.tagged")
				pathOri = r'testSet\\Origin\\' + re.sub(r'/', r'\\\\', line.split()[0][:-1] + ".xml.tagged")

			try:
				tree = ET.ElementTree(file=path)
				treeOri = ET.ElementTree(file=pathOri)
			except IOError:
				print("FILE NOT FOUND !!")
				print(path)
				continue

			root = tree.getroot()
			rootOri = treeOri.getroot()

			childsOri = [child.text for child in rootOri.getchildren()]
			childs = [child.text for child in root.getchildren()]



			if not (len(childsOri) == len(childs)):
				print("INCONSISTENT NUMBER OF CHILDS")

			for i in range(len(childs)):

				if childsOri[i] is not None and childs[i] is not None:
					A, B = TextAlignment([c.split()[0] for c in childsOri[i].split('\n') if c], [c.split()[0] for c in childs[i].split('\n') if c])
				else:
					raise Exception("There is a inconsistent amount of pages")

				for j in range(len(A)):
					
					if A[j] == B[j]:
						continue
					
					word_prob = 0
					countDiff = 0

					
					if A[j] == "°" or B[j] == "°" or (j+1 < len(A) and A[j+1] == "°"):
						continue
											
					for edit in alignWords(A[j], B[j]):
						if edit.split("|")[0] != edit.split("|")[1]:
							countDiff+=1
							print(edit, -log10(edits(edit)))
							

							if len(edit.split("|")[0]) == 3 or len(edit.split("|")[1]) == 3:
								print("KKKKKKKKKKKKKKKKK")



						word_prob+=log10(edits(edit))
					print(A[j], B[j],">>>>",countDiff,word_prob)
					print("---")
					
					x.append(-word_prob)
					y.append(countDiff)
					
							
					logProbs.append(word_prob)
					if -word_prob < 0:
						out.write(str(0) + '\n')
					else:
						out.write(str(-word_prob) + '\n')


			sys.stdout.write('\r' + str(count) + " / " + str(num_lines))
			sys.stdout.flush()
		print('\n')

		out.close()



		# Plot
		for i in range(len(x)):
			print(str(x[i])+"	"+str(y[i])+'\n')
		colors = "red"#(204,0,0)
		area = np.pi*6
		plt.scatter(x, y, s=area, c=colors, alpha=0.5)
		plt.title('Scatter plot EM logProbs')
		plt.xlabel('-logProb')
		plt.ylabel('number correction operations')
		plt.show()



		## WRITE TO FILE


		mean = sum( -key / float(len(logProbs)) for key in logProbs)

		variance = sum( ((-key - mean) ** 2) / len(logProbs) for key in logProbs)

		standardDeviation = sqrt(variance)

		print("NOISY EM")
		print("Mean:", mean)
		print("Variance:", variance)
		print("StandardDeviation:", standardDeviation)


	#LanguageModelNorvigUnigramHistogram()
	#ErrorModelNorvigUnigramHistogram(unigrams1, bigrams1, trigrams1)

	#LanguageModelNorvigBigramHistogram()
	#ErrorModelNorvigBigramHistogram(unigrams2, bigrams2, trigrams2)

	LanguageModelNoisyHistogram(LM)
	ErrorModelNoisyHistogram(EM)


def reTrainErrorModel(Unigrams, Bigrams):

	edits = ErrorModel()
	edits.setUnigrams(Unigrams)
	edits.setBigrams(Bigrams)


	count = 0
	num_lines = sum(1 for _ in open('/nethome/cklaus/Desktop/BachelorThesis/training_data.txt', "r", encoding="utf8"))
	trainFiles = open('/nethome/cklaus/Desktop/BachelorThesis/training_data.txt', "r", encoding="utf8")

	for line in trainFiles:
		count+=1
		if platform.system() == "Linux":
			
			path = '/nethome/cklaus/Desktop/BachelorThesis/CorrectedCorpus/' + line.split()[0][:-1] + ".xml.tagged"
			pathOri = getOriginPath(line)+".tagged"
		else:
			path = 'testSet\\NoisyChannel\\' + re.sub(r'/', r'\\\\', line.split()[0][:-1] + ".xml.tagged")
			pathOri = getOriginPath(line)+".tagged"


		try:
			tree = ET.ElementTree(file=path)
			treeOri = ET.ElementTree(file=pathOri)
		except IOError:
			print("FILE NOT FOUND !!")
			print(path)
			continue

		root = tree.getroot()
		rootOri = treeOri.getroot()

		childsOri = [child.text for child in rootOri.getchildren()]
		childs = [child.text for child in root.getchildren()]

		if not (len(childsOri) == len(childs)):
			print("INCONSISTENT NUMBER OF CHILDS")

		print(count)

		for i in range(len(childs)):

			if (childsOri[i] is None and childs[i] is None):
				continue

			if (childsOri[i] is not None and childs[i] is not None):
				
				A, B = TextAlignment([c.split()[0] for c in childsOri[i].split('\n') if c], [c.split()[0] for c in childs[i].split('\n') if c])
			else:
				raise Exception("There is a inconsistent amount of pages")

			for j in range(len(A)):
				if A[j] == "°" or B[j] == "°" or (j+1 < len(A) and A[j+1] == "°"):
					continue
					# TODO check wheter space insertion has a negative effect of the edits here -> should not be A[j]==° a seperate case?
				for edit in alignWords(A[j], B[j]):
					edits[edit] = edits.get(edit, 0) + 1
				if A[j] == B[j]:
					for c in getBigrams(A[j]):
						edits[c + "|" + c] = edits.get(c + "|" + c, 0) + 1
				
			


	'MISSING UNIGRAMS / BIGRAMS'

	dic = {}
	for edit in edits:
		spl = edit.split("|")[1]
		if spl in edits.getUnigrams() or spl in edits.getBigrams() :
			pass
		else:
			dic[spl] = dic.get(spl, 0) + edits[edit]
	uni = open("unigrams.count", "a")
	bi = open("bigrams.count", "a")
	for d in dic:
		if len(d) == 1:
			uni.write(d + "	" + str(dic[d]) + "\n")
			tmp = edits.getUnigrams()
			tmp.update({d: dic[d]})
			edits.setUnigrams(tmp)
		elif len(d) == 2:
			bi.write(d + "	" + str(dic[d]) + "\n")
			tmp = edits.getBigrams()
			tmp.update({d: dic[d]})
			edits.setBigrams(tmp)
	uni.close()
	bi.close()

	## solve problem with deletion at beginning
	summe = 0
	for key in edits:
		spl = key.split('|')[1]
		if spl == ">":
			summe += edits[key]
	edits.setNumberDeletionAtBeginning(summe)

	## WRITE TO FILE
	sorted_EM = sorted(edits, key=edits.get, reverse=True)
	out = open("edits.count", "w")

	for key in sorted_EM:
		out.write(key + "	" + str(edits[key]) + '\n')
	out.close()

	return edits
	
	
	
def train():



	lm,  unigrams, bigrams = buildLanguageModel(True)
	error_model = buildErrorModel(readRules(), unigrams, bigrams,True)

	#lm,  unigrams, bigrams = buildLanguageModelOLD(alreadyGenerated)


	global PREFIXES
	PREFIXES = set(w[:i] for w in lm.getVocabulary() for i in range(len(w) + 1))

	global blackList
	blackList = buildBlackList()

	#print("finished Training " + '\n')

	return lm, error_model


def reTrain():


	lm, unigrams, bigrams = buildLanguageModel(False,False)
	em = reTrainErrorModel(unigrams, bigrams)

	global PREFIXES
	PREFIXES = set(w[:i] for w in lm.getVocabulary() for i in range(len(w) + 1))

	global blackList
	blackList = buildBlackList()



	print("finished Re-Training")

	return lm, em



def correctCorpus(LM,EM):


	documents = open("/nethome/cklaus/Desktop/BachelorThesis/training_data.txt", "r", encoding="utf8")
	Vocabulary = LM.getVocabulary()


	num_cores = multiprocessing.cpu_count()


	data = [ (Vocabulary, LM, EM, name, False)  for name in documents] 

	p = multiprocessing.Pool(num_cores)
	p.starmap(correctDocument, data)


		



def optimizeLMWeight(LM, EM):


	global PRIOR_WEIGHT

	
	for i in range(20):
		
		for j in [1,-1]:
			if  i == 0 and j == -1:
				continue
			
			PRIOR_WEIGHT = 1.35 + j * 0.05 * i

			sys.stdout.write(str(PRIOR_WEIGHT)+"\t")
			sys.stdout.flush()
	
			correctionTestSet(LM, EM)
			testMetrics("<noisy>")


			LM()
			EM()
			editProbability()
			edProb()
			edits()




def optimizeEMSmooth(LM, EM):
	
	global SMOOTH_EM


	#for i in [0.0000000000001,0.00000000000001,0.000000000000001,0.0000000000000001,0.00000000000000001,0.000000000000000001]:


	for i in range(20):
		
		for j in [1,-1]:
			if  i == 0 and j == -1:
				continue
			

			SMOOTH_EM = 0.0001 * (10)**(i*j)


			sys.stdout.write(str(SMOOTH_EM)+"\t")
			sys.stdout.flush()		

			correctionTestSet(LM, EM)
			testMetrics("<noisy>")


			LM()
			EM()
			editProbability()
			edProb()
			edits()



def optimizeLMSmooth(LM, EM):

	global SMOOTH_LM


	for i in range(20):
	
		for j in [1,-1]:
			if  i == 0 and j == -1:
				continue
			

			SMOOTH_LM = 1e-06 * (10)**(i*j)


			sys.stdout.write(str(SMOOTH_LM)+"\t")
			sys.stdout.flush()		

			correctionTestSet(LM, EM)
			testMetrics("<noisy>")


			LM()
			EM()
			editProbability()
			edProb()
			edits()

		


def optimizeOOVESTIMATION(LM, EM):


	global OOV


	for i in range(20):
		
		for j in [1,-1]:
			if  i == 0 and j == -1:
				continue
		
			OOV = 13500000000 + i * j *  50000000

			sys.stdout.write(str(OOV)+"\t")
			sys.stdout.flush()


			correctionTestSet(LM, EM)
			testMetrics("<noisy>")

			LM()
			EM()
			editProbability()
			edProb()
			edits()



def reFormatXMLFiles():
	

	for t in open("/nethome/cklaus/Desktop/BachelorThesis/training_data.txt", "r", encoding="utf8"):


		try:
			path = "/nethome/cklaus/Desktop/BachelorThesis/CorrectedCorpus/" + t.split()[0][:-1] + ".xml.tagged"
			tree = ET.ElementTree(file=path)
		except IOError:
			print("FILE NOT FOUND !!")
		except FileNotFoundError:
			print("FILE NOT FOUND !!")
		#except:



			datei = open(path).read()
							
			datei = re.sub(r'&amp\tNN\t&amp\n;\t:\t;\n', r'&amp;\tNN\t&amp;\n', datei)
			datei = re.sub(r'&amp\tNNS\t&amp\n;\t:\t;\n', r'&amp;\tNNS\t&amp;\n', datei)
			datei = re.sub(r'&#167\tNNS\t&#167\n;\t:\t;\n', r'&#167;\tNNS\t&#167;\n', datei)
			datei = re.sub(r'&#167\tNN\t&#167\n;\t:\t;\n', r'&#167;\tNN\t&#167;\n', datei)
			datei = re.sub(r'&#167\tJJ\t&#167\n;\t:\t;\n', r'&#167;\tJJ\t&#167;\n', datei)
			datei = re.sub(r'&#167\tJJ\t&#167\n;\t:\t;\n', r'&#167;\tJJ\t&#167;\n', datei)
	
			datei = re.sub(r'&gt\tNNS\t&gt\n;\t:\t;\n', r'&gt;\tNNS\t&gt;\n', datei)
			datei = re.sub(r'&gt\tNN\t&gt\n;\t:\t;\n', r'&gt;\tNN\t&gt;\n', datei)


			datei = re.sub(r'&lt\tNNS\t&lt\n;\t:\t;\n', r'&lt;\tNNS\t&lt;\n', datei)
			datei = re.sub(r'&lt\tNN\t&lt\n;\t:\t;\n', r'&lt;\tNN\t&lt;\n', datei)

			datei = re.sub(r'&lt\tVVD\t&lt\n;\t:\t;\n', r'&lt;\tVVD\t&lt;\n', datei)
			datei = re.sub(r'&gt\tVVD\t&gt\n;\t:\t;\n', r'&gt;\tVVD\t&gt;\n', datei)
			datei = re.sub(r'&amp\tVVD\t&amp\n;\t:\t;\n', r'&amp;\tVVD\t&amp;\n', datei)
			datei = re.sub(r'&amp\tVVZ\t&amp\n;\t:\t;\n', r'&amp;\tVVZ\t&amp;\n', datei)
			datei = re.sub(r'&lt\tVVZ\t&lt\n;\t:\t;\n', r'&lt;\tVVZ\t&lt;\n', datei)
			datei = re.sub(r'&gt\tVVZ\t&gt\n;\t:\t;\n', r'&gt;\tVVZ\t&gt;\n', datei)
	
			datei = re.sub(r'&gt\tJJ\t&gt\n;\t:\t;\n', r'&gt;\tJJ\t&gt;\n', datei)
			datei = re.sub(r'&lt\tJJ\t&lt\n;\t:\t;\n', r'&lt;\tJJ\t&lt;\n', datei)
			datei = re.sub(r'&amp\tJJ\t&amp\n;\t:\t;\n', r'&amp;\tJJ\t&amp;\n', datei)


			datei = re.sub(r'&gt\tNP\t&gt\n;\t:\t;\n', r'&gt;\tNP\t&gt;\n', datei)
			datei = re.sub(r'&lt\tNP\t&lt\n;\t:\t;\n', r'&lt;\tNP\t&lt;\n', datei)
			datei = re.sub(r'&amp\tNP\t&amp\n;\t:\t;\n', r'&amp;\tNP\t&amp;\n', datei)

			datei = re.sub(r'&gt\tRB\t&gt\n;\t:\t;\n', r'&gt;\tRB\t&gt;\n', datei)
			datei = re.sub(r'&lt\tRB\t&lt\n;\t:\t;\n', r'&lt;\tRB\t&lt;\n', datei)
			datei = re.sub(r'&amp\tRB\t&amp\n;\t:\t;\n', r'&amp;\tRB\t&amp;\n', datei)

			datei = re.sub(r'&gt\tVV\t&gt\n;\t:\t;\n', r'&gt;\tVV\t&gt;\n', datei)
			datei = re.sub(r'&lt\tVV\t&lt\n;\t:\t;\n', r'&lt;\tVV\t&lt;\n', datei)
			datei = re.sub(r'&amp\tVV\t&amp\n;\t:\t;\n', r'&amp;\tVV\t&amp;\n', datei)


			datei = re.sub(r'&gt\tVVG\t&gt\n;\t:\t;\n', r'&gt;\tVVG\t&gt;\n', datei)
			datei = re.sub(r'&lt\tVVG\t&lt\n;\t:\t;\n', r'&lt;\tVVG\t&lt;\n', datei)
			datei = re.sub(r'&amp\tVVG\t&amp\n;\t:\t;\n', r'&amp;\tVVG\t&amp;\n', datei)


			datei = re.sub(r'&gt\tVVP\t&gt\n;\t:\t;\n', r'&gt;\tVVP\t&gt;\n', datei)
			datei = re.sub(r'&lt\tVVP\t&lt\n;\t:\t;\n', r'&lt;\tVVP\t&lt;\n', datei)
			datei = re.sub(r'&amp\tVVP\t&amp\n;\t:\t;\n', r'&amp;\tVVP\t&amp;\n', datei)


			datei = re.sub(r'&gt\tVVN\t&gt\n;\t:\t;\n', r'&gt;\tVVN\t&gt;\n', datei)
			datei = re.sub(r'&lt\tVVN\t&lt\n;\t:\t;\n', r'&lt;\tVVN\t&lt;\n', datei)
			datei = re.sub(r'&amp\tVVN\t&amp\n;\t:\t;\n', r'&amp;\tVVN\t&amp;\n', datei)



			f = open(path, "w")
			f.write(datei)
			f.close()
			
	
		


def readArguments():
	
	parser=argparse.ArgumentParser(description='Noisy Channel Spell Checker'
									#, usage='%(prog)s [options]
									,formatter_class=argparse.ArgumentDefaultsHelpFormatter
									)

	## CORRECTION
	parser.add_argument("-c", '--correct',  type=str ,metavar="INPUT", nargs='*', help='Text that is supposed to be corrected by the spell checker. You can enter one or more files, multiple directories or direct input. Directories are recursively traversed ')
	parser.add_argument("-ow", '--overwrite', action='store_true', help='If set, all the selected documents are overwritten by its correction.')
	parser.add_argument("-o", '--output', nargs=1, default = "", help='Determine where to store the corrected files (per default: location of input data)')
	
	
	## TRAINING
	parser.add_argument("--arpa", nargs=1, default="LM.arpa", metavar="LM",  help='ARPA file to instantiate the language model, skips LM training')
	parser.add_argument("-lm","--languagemodel", nargs=1, default="LM.arpa",  metavar="LM", help=' Filename to determine where to store the trained, arpa-formated language model. ')
	parser.add_argument("-tr","--train", nargs="+",metavar="DATA",  help='Training files to train a language model. You can enter file(s) or entire folder(s).')
	
	
	group2 = parser.add_mutually_exclusive_group()
	group2.add_argument('-ocr', action='store_true', help='Use tool to post-process OCR noise text', default=True)
	group2.add_argument('-typo', action='store_true', help='Use tool to correct texts containing typos.')
	
	
	## PARAMETERS
	parser.add_argument("--lmweight", nargs=1, default=1.25,  metavar="WEIGHT",type=float,   help='numeric value w that weights the language model P(c)^w')
	parser.add_argument("--order", nargs=1, type=int, default=2,  metavar="ORDER",  help='Order of generated language model.')
	parser.add_argument("--error_model_smooth", nargs=1, default=0.0001,  metavar="PSEUDOCOUNT",type=float,   help='pseudocount for laplace smoothing of the error model probabilities')
	
	
	## ADDITIONAL OPTIONS
	parser.add_argument("-sw", "--stopwords", nargs="+", metavar="STOPWORDS",   help='list of stopwords being ignored during correction, *.txt file | direct input')
	parser.add_argument("-v", "--version",action='store_true', help='Prints version of the Noisy Channel Spell Checker.')

	group = parser.add_mutually_exclusive_group()
	group.add_argument('-q', '--quiet', action='store_true', help=' Suppress printouts.')
	group.add_argument('-vb','--verbose', action='store_true', help='Print verbose.')
	group.add_argument('--skip_html', action='store_true', help='Ignore internal structure of HTML or XML files.')
	
	## TEST
	parser.add_argument("-te","--test", nargs=2,  help='Evaluates the tool on a selected pair of files. 2 Arguments: A file with misspellings included,  proper correction to examine (default: Royal Society Corpus)')
	parser.add_argument('--royal',action='store_true',  help='Evaluates the spell checker on the sample documents from the Royal Society Corpus')
	parser.add_argument('-ppl','--perplexity', nargs="*", help='Computes the Perplexity measure for given file(s) | folder(s)',default="groundTruth.txt")


	args=parser.parse_args()
	return args

def processArguments(args):

	# workaround if argparse has set the default value
	def checkDefault(o):
		if isinstance(o, (list,)):
			return o[0]
		else:
			return o

	#print(args)
	
	## PARAMETERS
	
	global PRIOR_WEIGHT 
	PRIOR_WEIGHT= checkDefault(args.lmweight) 
	
	global SMOOTH_EM
	SMOOTH_EM = checkDefault(args.error_model_smooth)
	
	global N_GRAM
	N_GRAM =  checkDefault(args.order)
	
	## VERSION
	if args.version:
		printVersion()
	
	
	## META
	
	global OVERWRITE
	OVERWRITE = checkDefault(args.overwrite)
	
	global DEST 
	DEST = checkDefault(args.output)
	if DEST:
		DEST = DEST if DEST and DEST[-1] == "/" else DEST+"/"


	### TRAINING

	global OCR
	if args.typo:
		OCR = False
	elif args.ocr:
		OCR=True


	
	if args.arpa:
		pass
	
	
	elif args.train != None:
		
		files = []
		directories = []
		
		for data in args.train:
			if os.path.isfile(data):
				files.append(data)
				#FILES
			elif os.path.isdir(data):
				directories.append(data)
				#DIRECTORY
			else:
				print(data, "is no file or directory")

		
		print(files)
		print(directories)
	
	else:
		pass # TODO default *arpa

	LM=None
	EM=None



	### CORRECTION
	
	files = []
	directories = []
	tokens = []
	prompt = False

	if args.correct != None:

		if len(args.correct) > 0:
			for data in args.correct:
				if os.path.isfile(data):
					files.append(data)
					#FILES
				elif os.path.isdir(data):
					directories.append(data)
					#DIRECTORY
				else:
					tokens.append(data)
					#DIRECT INPUT
		else:
			prompt = True	#PROMPT

	
		
		
		
	return LM,EM,{"files":files, "directories":directories, "tokens":tokens, "prompt":prompt}
		
		
		

def printVersion():
	print("Noisy Channel Spell Checker   " + VERSION)
	



def correctionPrompt(LM, EM):
	print("Type text and submit with RETURN. Type 'quit()' when you are done.")
	inputText = None  
	while True:    
		inputText = input(">>>  ")
		if inputText == "quit()":
			break
		else:
			print(correctPlainText(LM, EM, inputText))
	return


def process_correction_input(input, LM, EM):
	
	## PROMPT
	if input["prompt"]:
		correctionPrompt(LM,EM)
		return
	
	## CORRECT DIRECT INPUT
	print(correctPlainText(LM, EM, input["tokens"]))
	
	## CORRECT FILE
	for file in input["files"]:
		correctFile(LM, EM, file)
	
	## CORRECT FILES IN FOLDER
	# traverse recursively folder 
	for folder in input["directories"]:
		try:
			for root, subdirs, files in os.walk(folder):
				for f in files:
					if not ".corrected" in f:
						if root[-1] != "/":
							root+="/"
						#correctFile(LM, EM, root + f, dfksilfjaoisdjfoisdjafi jip    )
		except IOError:
			print("Did really mean {0}?".format(folder))


		
def correctFile(LM, EM, file_name, new_dest = ""):
	try:
		with open(file_name) as file:
			
			# TODO READ WHOLE FILE GOOD?
			
			data = file.read()
		
			## TODO MIT DEN DREI ZEILEN UNTER KANN MAN DIE DATEI DIREKT ÜBERSCHREIBEN:
			## VORSICHT: XML TAGS WERDEN AUCH ÜBERSCHREIBEN


			file.seek(0)
			file.truncate()
			file.write("Und weg ist alles!")




		
			# create output document
			new_file_name = file_name
			if not OVERWRITE:
				new_file_name = file_name[::-1].replace("."[::-1],".corrected."[::-1], 1)[::-1]
			if new_file_name.endswith("."):
				new_file_name = new_file_name[:-1]
			
			print("->",new_file_name)
			with open(new_file_name, "w") as fileOut:
				fileOut.write(correctPlainText(LM, EM, data))
	
	except IOError as e:
		print("File {0} not found.".format(file_name))
		
		


def main(alreadyGenerated):


	LM, EM = train()	


	#TODO DOCKER FILE

		
	args = readArguments()
	LM2, EM2, correction_input = processArguments(args)
			
	
	process_correction_input(correction_input, LM, EM)
	
	print()
	return
	
	
	if not args.correct:
		correctionPrompt(LM,EM)
	
	#LM, EM = reTrain()

	
	
	
	#correctCorpus(LM,EM)
	correctionTestSet(LM, EM)

	
	#for algo in ["<noisy>", "<rules>", "<norvigUni>", "<norvigBi>"]:
	for algo in ["<noisy>"]:
		print(">>> " + algo)
		testMetrics(algo)
	#computePerplexity()








	#summe = 0
	#inte = "e"

	#for u in EM.getUnigrams():
		#summe+=EM(u+"|"+inte)
	#for b in EM.getBigrams():
		#summe+=EM(b+"|"+inte)
	#for t in EM.getTrigrams():
		#summe+=EM(t+"|"+inte)
		
	#print(summe)
	#return

	
	#global UNK_ESTIMATION
	#UNK_ESTIMATION = 1
	#Vocab = {k: LM.getVocabulary()[k] for k in LM.getVocabulary()}
	#Vocab.update({"<unk>":1})
	#summe = 10**LM("<s>") + 10**LM("</s>")
	
	#for v in Vocab:
		#summe+=10**LM(v)
		
	#print(summe)
	
	
	#prev="the"
	#Vocab = {k: LM.getVocabulary()[k] for k in LM.getVocabulary()}
	#Vocab.update({"<unk>":1})
	
	#summe=0
	#for key in LM:
		#l=key.split(" ")
		#if len(l) == 2 and l[0] == prev:
			#summe+=10**LM(l[1],prev)
			
			#try:
				#del Vocab[l[1]]
			#except:
				#pass
	#for v in Vocab:
		#summe+=10**LM(v,prev)
	
	#print(summe)
	
	
	#prev2="the"
	#prev1 = "SCHUBUDUBIDUBIDU"
	#Vocab = {k: LM.getVocabulary()[k] for k in LM.getVocabulary()}
	#Vocab.update({"<unk>":1})
	
	#summe=0
	#for key in LM:
		#l=key.split(" ")
		#if len(l) == 3 and l[0] == prev1 and l[1] == prev2:
			#summe+=10**LM(l[2],prev2,prev1)
			
			#try:
				#del Vocab[l[2]]
			#except:
				#pass
	#for v in Vocab:
		#summe+=10**LM(v,prev2,prev1)
	
	#print(summe)
	
	
	#return


	






if __name__ == "__main__":
	sys.exit(main(True))








