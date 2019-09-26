
from __future__ import print_function
import builtins as __builtin__
from math import log10
import xml.etree.cElementTree as ET  # to read xml files
import re  # regular expressions
import sys
import codecs
import platform
import os
import subprocess  # to open shell inside this program flow
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import string
import argparse
from pathlib import Path
import warnings


VERSION = "1.0.31"
SMOOTH_LM = 1e-06
SMOOTH_EM = 0.0001
PRIOR_WEIGHT = 1.25

N_GRAM = 2
DISTANCE_LIMIT = 2
COUNT_THRESHOLD = 2

OOV = 13250000000

OVERWRITE = False
DESTINATION_DIR = "output/"
OCR = True
TARGET_LANGUAGE_MODEL = ""

QUIET = False
VERBOSE = False

EVALUATE_ROYAL_SOCIETY_CORPUS = False
DATA_DIR = 'data/'
SRILM_PATH = "/nethome/cklaus/tools/srilm/bin/i686-m64"
NUM_CORES = max(1,multiprocessing.cpu_count()-1)


CORRECT_RSC = False

# Suppress Warnings
warnings.filterwarnings("ignore")


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

            # this block is necessary to give functions with a history list a special treatment
            try:
                if isinstance(args[2], list):
                    args = (args[0], args[1], " ".join(args[2]))
            except IndexError:
                pass

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




def print(*args, **kwargs):
    """My custom print() function."""
    # Adding new arguments to the print function signature
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs
    if not QUIET:
        return __builtin__.print(*args, **kwargs)


PREFIXES = []
blackList = {}
stopwords = {}



class LanguageModelOLD(dict):

    Vocabulary = {}
    N = 0

    def __init__(self, newInstance):

        if not newInstance:
            for line in open(Path(os.path.join(DATA_DIR, "LanguageModel.count")) , "r", encoding="iso-8859-1"):
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

    vocab = {}


    def __init__(self, arpa_file=None):

        # Load LM with predefined arpa file
        if arpa_file:
            arpa = open(arpa_file, "r", encoding="iso-8859-1")


            while arpa.readline().strip() != "\\data\\":
                continue

            # Extract how many 1-grams, 2-grams,... there are
            numberNGrams = []
            for i in range(N_GRAM):
                numberNGrams.append( int(arpa.readline().strip().split("=")[1] ) )

            order_counter = 0
            while True:

                try:
                    line = arpa.readline().split()
                except:
                    raise ValueError("{0} is not well formated, crashed at line {1}.".format(arpa_file," ".join(line)))



                if line:
                    if line[0] == "\\end\\":
                        break

                    elif "-grams:" in line[0]:
                        order_counter += 1

                    elif order_counter < N_GRAM:
                        try:
                            self[ " ".join(line[1:order_counter+1])] = (float(line[0]), float(line[order_counter+1]))
                        except IndexError:
                            self[ " ".join(line[1:order_counter+1]) ] = (float(line[0]), )

                    elif order_counter == N_GRAM:
                        self[ " ".join(line[1:order_counter+1]) ] = (float(line[0]), )


    @memo2
    def __call__(self, current, history):



        # replace unknown tokens with <unk>

        if not current in self.vocab:
            if current == "<s>":
                return self["<s>"][0]
            if current == "</s>":
                return self["</s>"][0]

            current = "<unk>"

        # Solve Bug: If history is empty and word not in self
        if not history and current not in self:
            current = "<unk>"

        s = current

        history = [h for h in history.split(" ") if h]

        # replace unknown words in history with UNK token
        for i in range(len(history)):
            if history[i] not in self.vocab:




                history[i] = '<unk>'

        s = " ".join(history) + " " + s if history else s


        ###  d : (prob, backoff)  ###

        if s in self:

            # add some extra ballast if word is unknown
            unk = log10(1./OOV) if current == "<unk>" else 0

            return self[s][0] + unk

        # n-gram not seen: backoff
        else:
            return self(current, history[1:]) + self.backoff(history)

    def backoff(self, history):

        s = " ".join(history)

        try:
            return self.get(s)[1]
        except:
            return 0

    def setVocabulary(self, voc):
        self.vocab = voc

    def getVocabulary(self):
        return self.vocab






class ErrorModel(dict):

    _unigrams = {}
    _bigrams = {}


    def __init__(self, unigrams, bigrams, modus):

        self._unigrams = unigrams
        self._bigrams = bigrams

        editFile = open(Path(DATA_DIR, modus + '.txt'), "r", encoding="iso-8859-1")

        # read in edit file
        for line in editFile:
            line_list = line.split('\t')
            self[line_list[0]] = int(line_list[1])

            # ensures well-formated edit file
            assert(len(line_list) == 2)

        # Sometimes we see edits like >e|> which means the right side of the edit transformation is empty.
        # They need to be handled in a special case.

        summe = 0
        for key in self:
            if key.split('|')[1] == '>':
                summe += self[key]
        self._numberDeletionAtBeginning = summe


    @memo2
    def __call__(self, edit):

        if edit:
            denom = 0
            edit_freq = self.get(edit, 0)
            right_side = re.sub(">", "", edit.split('|')[1])

            # Example "e|e"
            if len(right_side) == 1:
                denom = self._unigrams.get(right_side, 0)
            #Example "e|ee"
            elif len(right_side) == 2:
                denom = self._bigrams.get(right_side, 0)

            # This happends from time to time e.g. >e|>
            elif len(right_side) == 0:
                denom = self._numberDeletionAtBeginning # * 100
            else:
                pass

            # Safety case: return a very small value
            if denom == 0:
                return 0.000000000000000001

            return (edit_freq + SMOOTH_EM) / float(denom + SMOOTH_EM * len(self))
        else:
            return 1

    def setUnigrams(self, unigrams):
        self.unigrams = unigrams

    def setBigrams(self, bigrams):
        self.bigrams = bigrams

    def getUnigrams(self):
        return self.unigrams

    def getBigrams(self):
        return self.bigrams



#def getOriginPath(line):
#    path = re.sub(r':', r'', 'data/Origin/' + line.split()[0] + '.xml')
#    if platform.system() != "Linux":
#        path = re.sub(r'/', r'\\', path)
#    return path


def getPath(line):
    path = re.sub(r':', r'', 'data/corpusTagged/' + line.split()[0] + '.xml.tagged')
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



def buildLanguageModel(arpa_file=None, files=[], TestSet = True):

    Unigrams = {}
    Bigrams = {}
    # Trigrams = {}

    Vocab = {}


    if arpa_file:

        LM = LanguageModel(arpa_file)

        for line in open(os.path.join(Path(DATA_DIR, "unigrams.count")), "r", encoding="iso-8859-1"):
            uniList = line.split('\t')
            Unigrams[uniList[0]] = int(uniList[1])

        for line in open(os.path.join(Path(DATA_DIR, "bigrams.count")), "r", encoding="iso-8859-1"):
            biList = line.split('\t')
            Bigrams[biList[0]] = int(biList[1])

        for line in open(os.path.join(Path(DATA_DIR, "vocabulary.count")), "r", encoding="iso-8859-1"):
            vocList = line.split('\t')
            Vocab[vocList[0]] = int(vocList[1])

        LM.setVocabulary(Vocab)

    elif files:

        corpus = open(os.path.join(Path(DATA_DIR, "corpus.txt")), "w", encoding="iso-8859-1")


        for f in files:

            if str(f).endswith('.xml'):
                path = Path(f)

                try:
                    tree = ET.ElementTree(file=path)
                except IOError:
                    print("FILE NOT FOUND ! " + str(f))
                    continue

                root = tree.getroot()

                for child in list(root):
                    if child.text is not None:

                        tokens = [c for c in child.text.split() if c]
                        for t in tokens:
                            # fill character unigrams
                            for uni in getUnigrams(t):
                                Unigrams[uni] = Unigrams.get(uni, 0) + 1

                            # fill character bigrams
                            for bi in getBigrams(t):
                                Bigrams[bi] = Bigrams.get(bi, 0) + 1

                            Vocab[t] = Vocab.get(t, 0) + 1

                        corpus.write(child.text + "\n")




            elif str(f).endswith('.xml.tagged'):

                path = Path(f)

                try:
                    tree = ET.ElementTree(file=path)
                except IOError:
                    print("FILE NOT FOUND ! " + str(f))
                    continue

                root = tree.getroot()

                for child in list(root):

                    if child.text is not None:

                        tokens = [c.split()[0] for c in child.text.split('\n') if c]
                        for t in tokens:
                            corpus.write(t+" ")
                            # fill character unigrams
                            for uni in getUnigrams(t):
                                Unigrams[uni] = Unigrams.get(uni, 0) + 1

                            # fill character bigrams
                            for bi in getBigrams(t):
                                Bigrams[bi] = Bigrams.get(bi, 0) + 1

                            Vocab[t] = Vocab.get(t, 0) + 1

                        corpus.write("\n")



            # normal text file
            else:
                with open(Path(f), "r") as file:
                    text = file.read()

                    tokens = [c for c in text.split() if c]
                    for t in tokens:
                    # fill character unigrams
                        for uni in getUnigrams(t):
                            Unigrams[uni] = Unigrams.get(uni, 0) + 1

                    # fill character bigrams
                        for bi in getBigrams(t):
                            Bigrams[bi] = Bigrams.get(bi, 0) + 1

                        Vocab[t] = Vocab.get(t, 0) + 1

                    corpus.write(text + "\n")

        corpus.close()


        global OOV
        OOV = 0

        # Only keep words with frequency over COUNT_TRESHOLD in vocabulary
        for key in [k for k, v in Vocab.items() if v]:
            if Vocab[key] < COUNT_THRESHOLD:
                del Vocab[key]
                OOV += 1

        #print("OOV", OOV)

        sorted_voc = sorted(Vocab, key=Vocab.get, reverse=True)
        voc = open(os.path.join(Path(DATA_DIR, "vocabulary.count")), "w")
        for key in sorted_voc:
            voc.write(key + "	" + str(Vocab[key]) + "\n")
        voc.close()



        sorted_uni = sorted(Unigrams, key=Unigrams.get, reverse=True)
        sorted_bigr = sorted(Bigrams, key=Bigrams.get, reverse=True)

        unigr = open(os.path.join(Path(DATA_DIR, "unigrams.count")), "w")
        for key in sorted_uni:
            unigr.write(key + "	" + str(Unigrams[key]) + '\n')
        unigr.close()

        bigr = open(os.path.join(Path(DATA_DIR, "bigrams.count")), "w")
        for key in sorted_bigr:
            bigr.write(key + "	" + str(Bigrams[key]) + '\n')
        bigr.close()


        # call SRILM
        subprocess.call(SRILM_PATH + "/ngram-count -vocab "+DATA_DIR+"/vocabulary.count   -order " + str(
            N_GRAM) + "   -no-eos -no-sos    -text "+DATA_DIR+"/corpus.txt  -unk  -write "+DATA_DIR+"/count" + str(N_GRAM) + ".count",
                        shell=True)

        print("created Count File")

        smooth = ""
        if N_GRAM != 1:
            for i in range(N_GRAM):
                smooth += " -kndiscount" + str(i + 1) + " "

        subprocess.call(SRILM_PATH + "/ngram-count   -vocab "+DATA_DIR+"/vocabulary.count   -order " + str(
            N_GRAM) + "  -unk -no-eos  -no-sos -read "+DATA_DIR+"/count" + str(N_GRAM) + ".count  -lm "+DATA_DIR+"/" +
                        TARGET_LANGUAGE_MODEL +" " + smooth, shell=True)

        print("created Language Model")

        LM = LanguageModel(os.path.join(Path(DATA_DIR, TARGET_LANGUAGE_MODEL)))
        LM.setVocabulary(Vocab)


    else:
        print("Something went wrong. Please declare sources to instantiate a language model.")

    return LM, Unigrams, Bigrams



def GroundTruthToTxt():

    num_lines = sum(1 for _ in open(os.path.join(Path(DATA_DIR, "test_data.txt")), "r", encoding="utf8"))

    testData = open(os.path.join(Path(DATA_DIR, "test_data.txt")), "r", encoding="utf8")
    count = 0

    corpus = open(os.path.join(Path(DATA_DIR, "groundTruth.txt")), "w")

    for file in testData:
        linebreaker = 0

        if platform.system() == "Linux":
            file = file.split()[0][:-1] + ".xml.tagged"
            pathGT = os.path.join(Path(DATA_DIR,'testSet','GroundTruth', file))
        else:
            file = re.sub(r'/', r'\\\\', file.split()[0][:-1] + ".xml.tagged")
            pathGT = os.path.join(Path(DATA_DIR, "testSet", "GroundTruth", file ))

        try:
            with open(pathGT, 'r', encoding="iso-8859-1") as f:
                tree = ET.parse(f)
        except IOError:
            print("File not found!")
            continue

        root = tree.getroot()

        for child in list(root):
            linebreaker += 1

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




# def buildErrorModel(rules, Unigrams, Bigrams, alreadyGenerated=False):
#     edits = ErrorModel()
#     edits.setUnigrams(Unigrams)
#     edits.setBigrams(Bigrams)
#
#     #if alreadyGenerated:
#     editFile = open("data/ocr.txt", "r", encoding="iso-8859-1")
#     for line in editFile:
#         lineList = line.split('\t')
#         edits[lineList[0]] = int(lineList[1])
#
#     ## solve problem with deletion at beginning
#     summe = 0
#     for key in edits:
#         spl = key.split('|')[1]
#         if spl == ">":
#             summe += edits[key]
#     edits.setNumberDeletionAtBeginning(summe)
#
#     else:
#     count = 0
#     num_lines = sum(1 for _ in open("data/training_data.txt", "r", encoding="utf8"))
#     allTitles = open("data/training_data.txt", "r", encoding="utf8")
#     rules_cpy = {k: rules[k] for k in rules}
#
#     for line in allTitles:
#
#     path = getOriginPath(line)
#
#     try:
#     tree = ET.ElementTree(file=path)
#     except IOError:
#     print("FILE NOT FOUND !!")
#     print(path)
#     continue
#
#     root = tree.getroot()
#
#     for child in list(root):
#
#     if child.text is not None:
#     childWordList = child.text.split()
#
#     for word in childWordList:
#
#     # WRONG WRITTEN WORD
#     if word in rules:
#
#     ### monitor which rules were already applied
#     try:
#     del rules_cpy[word]
#     except KeyError:
#     pass
#
#     if " " in rules[word][0]:
#     continue
#
#     for edit in alignWords(word, rules[word][0]):
#     edits[edit] = edits.get(edit, 0) + 1
#
#     # CORRECT WRITTEN WORD
#     else:
#     for c in list(word):
#     edits[c + "|" + c] = edits.get(c + "|" + c, 0) + 1
#     for c in getBigrams(word):
#     edits[c + "|" + c] = edits.get(c + "|" + c, 0) + 1
#     #for c in getTrigrams(word):
#     #	edits[c + "|" + c] = edits.get(c + "|" + c, 0) + 1
#
#
#
#
#     count += 1
#     sys.stdout.write('\r' + str(count) + " / " + str(num_lines))
#     sys.stdout.flush()
#     print('\n')
#
#     ### fidding around with unused rules
#     for w in rules_cpy:
#     for edit in [e for e in alignWords(w, rules_cpy[w][0])]:
#     edits[edit] = edits.get(edit, 0) + int(rules_cpy[w][1])
#
#     'MISSING UNIGRAMS / BIGRAMS'
#
#     dic = {}
#     for edit in edits:
#     spl = edit.split("|")[1]
#     if spl in edits.getUnigrams() or spl in edits.getBigrams():
#     pass
#     else:
#     dic[spl] = dic.get(spl, 0) + edits[edit]
#     uni = open("data/unigrams.count", "a", encoding="iso-8859-1")
#     bi = open("data/bigrams.count", "a", encoding="iso-8859-1")
#     #tri = open("data/trigrams.count", "a")
#     for d in dic:
#     if len(d) == 1:
#     uni.write(d + "\t" + str(dic[d]) + "\n")
#
#     tmp = edits.getUnigrams()
#     tmp.update({d: dic[d]})
#     edits.setUnigrams(tmp)
#     elif len(d) == 2:
#     bi.write(d + "\t" + str(dic[d]) + "\n")
#
#     tmp = edits.getBigrams()
#     tmp.update({d: dic[d]})
#     edits.setBigrams(tmp)
#     #else:
#     #tri.write(d + "\t" + str(dic[d]) + "\n")
#
#     #tmp = edits.getTrigrams()
#     #tmp.update({d: dic[d]})
#     #edits.setTrigrams(tmp)
#
#     uni.close()
#     bi.close()
#     #tri.close()
#
#     ## solve problem with deletion at beginning
#     summe = 0
#     for key in edits:
#     spl = key.split('|')[1]
#     if spl == ">":
#     summe += edits[key]
#     edits.setNumberDeletionAtBeginning(summe)
#
#     ## WRITE TO FILE
#     sorted_EM = sorted(edits, key=edits.get, reverse=True)
#     out = open("data/ocr.txt", "w", encoding="iso-8859-1")
#
#     for key in sorted_EM:
#     out.write(key + "	" + str(edits[key]) + '\n')
#     out.close()
#
#     return edits


def buildBlackList():


    global blackList

    # TODO check wheter this works. If so, then Remove TODO
    #return  blackList



    file = codecs.open(os.path.join(Path(DATA_DIR, "cleanDifferences.count")), "r", encoding="utf8")
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


# Source:  https://github.com/alevchuk/pairwise-alignment-in-python by Aleksandr Levchuk
# under GNU General Public License v3.0
def alignWords(a, b):

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


def assignPredecessors(history, current):
    global N_GRAM
    history.append(current)
    history = history[len(history) - N_GRAM + 1:]

    return history



def correct(Vocabulary, LM, EM, text, history):
    global blackList
    tag_blackList = ["SENT", ",", ":", "``", "''", "(", ")", "SYM"]
    tokens = [c.split() for c in text.split('\n') if c]

    i = -1
    for t in tokens:

        i += 1

        global stopwords

        if t[0] in stopwords:
            history = assignPredecessors(history, t)
            continue


        ## some tags do not need a correction, like interpunctions
        if t[1] in tag_blackList:
            history = assignPredecessors(history, t[0])
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

            res = LM(c, history) * PRIOR_WEIGHT + editProbability(EM, t[0], c)
            if res > maxi:
                maxi = res
                correctedWord = c

        history = assignPredecessors(history, t[0])

        t[0], t[1], t[2] = prefix + correctedWord + suffix, t[1], t[2]

    return "\n" + '\n'.join(['\t'.join(t) for t in tokens]) + "\n", history


def correct_plain_text(LM, EM, tokens, history=[]):



    vocabulary = LM.getVocabulary()

    if not isinstance(tokens, (list,)):
        tokens = [c.strip() for c in tokens.split(" ") if c]
    for i, t in enumerate(tokens):
        global stopwords
        if t in stopwords:
            history = assignPredecessors(history, t)
            continue

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
        for c in edits(vocabulary, EM, t):
            res = LM(c, history) * PRIOR_WEIGHT + editProbability(EM, t, c)
            if res > maxi:
                maxi = res
                corrected_word = c

        history = assignPredecessors(history, t)

        t = prefix + corrected_word + suffix
        tokens[i] = t

    return ' '.join(t for t in tokens), history


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

    history = []
    count = 0

    if TestSet:
        test = "testSet"
    else:
        test = ""


    path = Path(os.path.join(DATA_DIR, test, 'Origin', fileName.split()[0][:-1] + ".xml.tagged"))

    try:
        tree = ET.ElementTree(file=path)
    except IOError:
        print("FILE NOT FOUND ! " + fileName)
        return

    root = tree.getroot()

    for child in list(root):

        count += 1

        if child.text is not None:
            child.text, history = correct(Vocabulary, LM, EM, child.text, history)

    # quick and dirty method to reset cache after every file
    LM()
    EM()
    editProbability()
    edProb()
    edits()

    if TestSet:
        tree.write(re.sub(r'Origin', r'NoisyChannel', str(path)))
    else:
        tree.write(re.sub(r'Origin', r'CorrectedCorpus', str(path)))

    sys.stdout.write("[DONE] " + fileName)
    sys.stdout.flush()


def correctionTestSet(LM, EM):


    testFiles = open(os.path.join(Path(DATA_DIR, "test_data.txt")))
    Vocabulary = LM.getVocabulary()

    global NUM_CORES


    Parallel(n_jobs=NUM_CORES)(delayed(correctDocument)(Vocabulary, LM, EM, name) for name in testFiles)


def test_file(file):

    TP, TN, FP, FN = 0,0,0,0

    path_ori = Path(os.path.join(DATA_DIR, 'testSet/Origin/', file))
    pathGT = Path(os.path.join(DATA_DIR, 'testSet/GroundTruth/', file))
    path = Path(os.path.join(DATA_DIR, 'testSet/NoisyChannel/', file))


    try:
        treeOri = ET.ElementTree(file=path_ori)
        with open(pathGT, 'r', encoding="iso-8859-1") as f:
            treeGT = ET.parse(f)
        tree = ET.ElementTree(file=path)
    except IOError:
        print("FILE NOT FOUND!  " + file)

    rootOri = treeOri.getroot()
    rootGT = treeGT.getroot()
    root = tree.getroot()

    childsOri = [child.text for child in list(rootOri)]
    childsGT = [child.text for child in list(rootGT)]
    childs = [child.text for child in list(root)]

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

    sys.stdout.write("[TESTED] " + file + "\n")
    sys.stdout.flush()


    return (TP, TN, FP, FN, file)


def test_royal_society_corpus():
    TP, TN, FP, FN = 1, 1, 1, 1

    test_files = open(Path(DATA_DIR,"test_data.txt"))

    data = [ (file.split()[0][:-1] + ".xml.tagged",)  for file in test_files]

    global NUM_CORES
    p = multiprocessing.Pool(NUM_CORES)

    for v in [ (i[0],i[1],i[2],i[3],i[4]) for i in p.starmap(test_file, data)]:

        TP+=v[0]
        TN+=v[1]
        FP+=v[2]
        FN+=v[3]



    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    f_score = (2 * precision * recall) / float(precision + recall)

    sys.stdout.write("Precision" + "\t" + str(precision) + "\n")
    sys.stdout.write("Recall" + "\t" + str(recall) + "\n")
    sys.stdout.write("F1-Score" + "\t" + str(f_score) + "\n")

    sys.stdout.flush()


    return precision, recall, f_score


def compute_perplexity(arguments):
    # requirement:  textfile  +  arpa file
    subprocess.call(SRILM_PATH + "/ngram -lm "+ str(Path(arguments[1])) +
                    " -unk -ppl " + str(Path(arguments[0])), shell=True)



def correctCorpus(LM,EM):


    documents = open( os.path.join(Path(DATA_DIR, "training_data.txt")), "r", encoding="utf8")
    Vocabulary = LM.getVocabulary()


    global NUM_CORES

    data = [ (Vocabulary, LM, EM, name, False) for name in documents]


    p = multiprocessing.Pool(NUM_CORES)
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
            test_royal_society_corpus()


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
            test_royal_society_corpus()


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
            test_royal_society_corpus()


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
            test_royal_society_corpus()

            LM()
            EM()
            editProbability()
            edProb()
            edits()





def readArguments():

    parser=argparse.ArgumentParser(description='Noisy Channel Spell Checker'
                                   #, usage='%(prog)s [options]
                                   ,formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                   )

    ## CORRECTION
    parser.add_argument("-c", '--correct',  type=str ,metavar="INPUT", nargs='*', help='Text that is supposed to be corrected by the spell checker. You can enter one or more files, multiple directories or direct input. Directories are recursively traversed ')
    parser.add_argument("-ow", '--overwrite', action='store_true', help='If set, all the selected documents are overwritten by its correction.')
    parser.add_argument("-o", '--output', default = "output/", help='Determine where to store the corrected files (per default: location of input data)')
    parser.add_argument("-d", '--data', default = "data/", help='Determine the location of the data like auxiliary files or training documents')


    ## TRAINING
    parser.add_argument("--arpa",  metavar="LM",  help='ARPA file to instantiate the language model, skips LM training')
    parser.add_argument("-lm", "--languagemodel", default="LM2.arpa",  metavar="LM", help=' Filename to determine where to store the trained, arpa-formated language model. ')
    parser.add_argument("-tr", "--train", nargs="+",metavar="DATA",  help='Training files to train a language model. You can enter file(s) or entire folder(s).')


    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('-ocr', action='store_true', help='Use tool to post-process OCR noise text', default=True)
    group2.add_argument('-typo', action='store_true', help='Use tool to correct texts containing typos.')


    ## PARAMETERS
    parser.add_argument("--lmweight", default=1.25,  metavar="WEIGHT",type=float,   help='numeric value w that weights the language model P(c)^w')
    parser.add_argument("--order", type=int, default=2,  metavar="ORDER",  help='Order of generated language model.')
    parser.add_argument("--error_model_smooth",  default=0.0001,  metavar="PSEUDOCOUNT",type=float,   help='pseudocount for laplace smoothing of the error model probabilities')


    ## ADDITIONAL OPTIONS
    parser.add_argument("-sw", "--stopwords", nargs="+", metavar="STOPWORDS",   help='list of stopwords being ignored during correction, *.txt file | direct input')
    parser.add_argument("-v", "--version",action='store_true', help='Prints version of the Noisy Channel Spell Checker.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-q', '--quiet', action='store_true', help=' Suppress printouts.')
    group.add_argument('-vb','--verbose', action='store_true', help='Print verbose.')


    ## TEST
    parser.add_argument('-te', '--test',action='store_true',  help='Evaluates the spell checker on the sample documents from the Royal Society Corpus')
    parser.add_argument('--royal',action='store_true',  help='Correct the entire Royal Society Corpus')
    parser.add_argument('-ppl','--perplexity', nargs=2, help='Computes the Perplexity measure for given file and language model')
    parser.add_argument("--num_cores","-cores", type=int, default=1,  metavar="CORES",  help='Number of cores that can be used for computations, default: N - 1  ')


    args=parser.parse_args()

    # DEVELOPMENT: output arguments
    #for k in args.__dict__:
    #    if args.__dict__[k] is not None:
    #        print(k, ">", args.__dict__[k])
    #    else:
    #        print(k, "None")

    return args


def process_arguments(args):


    #print(args)

    #for k, value in args._get_kwargs():
    #    if value is not None:
    #        print(">>",k, value)



    ## PARAMETERS

    global PRIOR_WEIGHT
    PRIOR_WEIGHT= args.lmweight

    global SMOOTH_EM
    SMOOTH_EM = args.error_model_smooth

    global N_GRAM
    N_GRAM = args.order

    ## VERSION
    if args.version:
        printVersion()
        sys.exit(0)



    ## META

    global DESTINATION_DIR
    DESTINATION_DIR = os.path.join(args.output)
    # safely check for existence and create output folder
    if not os.path.exists(Path(os.path.expanduser(DESTINATION_DIR))):
        os.makedirs(Path(os.path.expanduser(DESTINATION_DIR)))

    global DATA_DIR
    DATA_DIR = Path(os.path.expanduser(os.path.join(args.data)))



    global OVERWRITE
    OVERWRITE = args.overwrite

    # if overwrite option has been set we do not need an output folder.
    if OVERWRITE:
        DESTINATION_DIR = ''

    ### TRAINING


    global stopwords
    # Check whether stopwords were set
    if args.stopwords is not None:
        # Iterate over stopword instances
        for stop in args.stopwords:
            # If stopword is a file then open it and extract all tokens
            if os.path.isfile(stop):
                for stop_token in open(stop).read().split():
                    stopwords[stop_token] = 0
            # No file, direct input of stopwords
            else:
                stopwords[stop] = 0



    global TARGET_LANGUAGE_MODEL
    TARGET_LANGUAGE_MODEL = args.languagemodel

    global QUIET
    QUIET = args.quiet

    global VERBOSE
    VERBOSE = args.verbose


    global EVALUATE_ROYAL_SOCIETY_CORPUS
    EVALUATE_ROYAL_SOCIETY_CORPUS = args.test

    global CORRECT_RSC
    CORRECT_RSC = args.royal

    global NUM_CORES
    NUM_CORES = min(NUM_CORES, args.num_cores)



    if args.languagemodel is not None and os.path.isfile(Path(args.languagemodel)) and args.train is None:
        LM, unigrams, bigrams = buildLanguageModel(arpa_file=os.path.join(Path(args.languagemodel)))

    elif args.train is not None:

        file_container = []

        for data in args.train:
            # collect all training files
            if os.path.isfile(data):
                file_container.append(data)
            # collect all training files in training directories
            elif os.path.isdir(data):
                for root, subdirs, files in os.walk(data):
                    for f in files:
                        file_container.append(Path(root, f))
            # in any other case reject input
            else:
                print(data, "is neither a file nor a directory")


        print("Files", file_container)

        LM, unigrams, bigrams = buildLanguageModel(files=file_container)

    # Default: 2-gram Language Model of
    else:

        LM, unigrams, bigrams = buildLanguageModel(arpa_file=os.path.join(Path(DATA_DIR, "LM2.arpa")))



    # Train Error Model

    global OCR
    if args.typo:
        OCR = False
        EM = ErrorModel(unigrams, bigrams, 'typo')
    elif args.ocr:
        OCR = True
        EM = ErrorModel(unigrams, bigrams, 'ocr')

    # Additional Recourses

    global PREFIXES
    PREFIXES = set(w[:i] for w in LM.getVocabulary() for i in range(len(w) + 1))

    global blackList
    blackList = buildBlackList()

    print("finished Training " + '\n')





    if args.perplexity:
        compute_perplexity(args.perplexity)


    ### CORRECTION

    files = []
    tokens = []
    prompt = False


    if args.correct is not None:

        # Give files or strings as input
        if len(args.correct) > 0:
            for data in args.correct:
                data = Path(os.path.expanduser(data))
                # Extract files
                if os.path.isfile(data):
                    files.append(data)
                # Extract directories
                elif os.path.isdir(data):
                    for root, subdirs, dir_files in os.walk(data):
                        for f in dir_files:
                            files.append(Path(root, f))
                # Direct Token input
                else:
                    tokens.append(data)
        #If only -c/--correct is given, open prompt environment
        else:
            prompt = True

    correction_input = {"files": files, "tokens": tokens, "prompt": prompt}
    return LM, EM, correction_input





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
            print(correct_plain_text(LM, EM, inputText, [])[0])
    return


def process_correction_input(LM, EM, input):


    ## PROMPT
    if input["prompt"]:
        correctionPrompt(LM, EM)

    ## CORRECT DIRECT INPUT
    print(correct_plain_text(LM, EM, input["tokens"], [])[0])

    ## CORRECT FILE
    for file in input["files"]:
        correctFile(LM, EM, file)


    # correct Royal Society Corpus test set
    global EVALUATE_ROYAL_SOCIETY_CORPUS
    if EVALUATE_ROYAL_SOCIETY_CORPUS:
        print("Testing the model on the Royal Society Corpus test set")
        correctionTestSet(LM, EM)
        test_royal_society_corpus()

    global CORRECT_RSC
    if CORRECT_RSC:
        print("Start correcting the Royal Society Corpus")
        correctCorpus(LM, EM)





def correctFile(LM, EM, file_name):

        global DESTINATION_DIR
        data = open(file_name).read()
        if not data:
            return

        # Generate new document name
        new_file_name = str(file_name)
        # Separates file endings and initial dots like '.\file.txt'
        splitted_file_name = [comp for comp in new_file_name.split(".") if comp]

        if not OVERWRITE:
            # label appears in the file name of the corrected file
            label = "" if '_corrected' in splitted_file_name[0] else '_corrected'

            new_file_name = (''.join('.' + n for n in splitted_file_name[:-1]) + label + "." + splitted_file_name[-1])[1:]


            if new_file_name[0] == "/" or new_file_name[0] == "\\":
                new_file_name = new_file_name[1:]



            target_path, target_filename = os.path.split(new_file_name)
            target = Path(os.path.join(Path(os.path.expanduser(DESTINATION_DIR)), target_filename))


            if not os.path.exists(Path(os.path.expanduser(DESTINATION_DIR))):
                os.makedirs(Path(os.path.expanduser(DESTINATION_DIR)))


            # special file types: XML and annotated XML (verticalized)
            if str(file_name).endswith(".xml") or str(file_name).endswith(".xml.tagged"):

                try:
                    with open(file_name, 'r', encoding="iso-8859-1") as f:
                        tree = ET.parse(f)
                except IOError:
                    print("File not found!", file_name)
                    return

                root = tree.getroot()
                history = []

                for child in list(root):

                    if child.text is not None:
                        # distinguish between XML TAGGED and XML
                        if str(file_name).endswith(".xml.tagged"):
                            child.text, history = correct(LM.getVocabulary(), LM, EM, child.text, history)
                        elif str(file_name).endswith(".xml"):
                            child.text, history = correct_plain_text(LM, EM, child.text, history)

                tree.write(target)

            # ordinary text file
            else:
                with open(target, "w") as fileOut:
                    fileOut.write(correct_plain_text(LM, EM, data, [])[0])



        # Do not overwrite
        else:
            file_name = new_file_name
            # special file types: XML and annotated XML (verticalized)
            if str(file_name).endswith(".xml") or str(file_name).endswith(".xml.tagged"):

                try:
                    with open(file_name, 'r', encoding="iso-8859-1") as f:
                        tree = ET.parse(f)
                except IOError:
                    print("File not found!", file_name)
                    return

                root = tree.getroot()
                history=[]

                for child in list(root):

                    if child.text is not None:
                        # distinguish between XML TAGGED and XML
                        if str(file_name).endswith(".xml.tagged"):
                            child.text, history = correct(LM.getVocabulary(), LM, EM, child.text, history)
                        elif str(file_name).endswith(".xml"):
                            child.text, history = correct_plain_text(LM, EM, child.text, history)

                tree.write(file_name)

            #ordinary text file
            else:
                open(file_name, 'w').write(correct_plain_text(LM, EM, data, [])[0])



def activatePrompt(args):
    if args.correct is None and not args.test and not args.royal:
        return True
    else:
        return False

def main():

    args = readArguments()

    LM, EM, correction_input = process_arguments(args)

    process_correction_input(LM, EM, correction_input)

    print()

    if activatePrompt(args):
        correctionPrompt(LM, EM)

    return





if __name__ == "__main__":
    main()








