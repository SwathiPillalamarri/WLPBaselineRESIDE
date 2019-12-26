# converts protocols into correct format before preprocessing for RESIDE training
import re
import json
import sys
from os import listdir
from os.path import isfile, join
from pycorenlp import *
import pprint
import time

# create training data in format of raw riedel dataset from RESIDE model
# only for inter-AP relations

########################### parse protocols, return information about sentences/relations/entities
def parseProtocol(txtfile, annfile, protnum):
    print('filenum: ' + protnum)
    sentences = parseTxtFile(txtfile, protnum)
    entities, erels, rrels = parseAnnotationFile(sentences, annfile, protnum)
    #print(rrels)
    #print('Parsed: ' + protnum)
    return sentences, entities, erels, rrels

def parseTxtFile(txtfile, protnum):
    sentences = []
    curr_char = 0
    for line in txtfile.readlines():
        start_char = curr_char
        end_char = curr_char + len(line) # excluding end char
        sentences.append({"start_index": start_char, "end_index": end_char, "sent": line.strip()})
        curr_char = end_char
    return sentences

def parseAnnotationFile(sentences, annfile, protnum):
    entities = {}    # start id with T in annotation file
    erels = {"actions": {}, "rels": {}}      # start id with E in annotation file
    rrels = {}      # start id with R in annotation file

    # read in all entities first
    for line in annfile.readlines():
        if line[0]=="T":  # parse "T" id's in file
            id, entJSON = parseEntities(line, protnum, sentences)
            entities[id] = entJSON

    # then read in relations
    annfile.seek(0)
    for line in annfile.readlines():
        if line[0]=="E":  # parse "E" id's in file
            parseERels(entities, erels, line, protnum)
        elif line[0]=="R":  # parse "E" id's in file
            id, rrelJSON = parseRRels(entities, erels, line, protnum)
            rrels[id] = rrelJSON

    # find sentence numbers for relations on next pass
    removeFromERels = []    # keys in erels where relations are cross sentence
    for rel in erels['rels']:
        arg1 = erels['rels'][rel]['arg1']
        arg2 = erels['rels'][rel]['arg2']
        sent_num = arg2SentNum(rel, entities, erels, arg1, arg2, protnum)
        if sent_num==-1:
            removeFromERels.append(rel)
        else:
            erels['rels'][rel]['sent_num'] = sent_num

    for rel in removeFromERels:
        del erels['rels'][rel]

    removeFromRRels = []    # keys in rrels where relations are cross sentence
    for rel in rrels:
        arg1 = rrels[rel]['arg1']
        arg2 = rrels[rel]['arg2']
        sent_num = arg2SentNum(rel, entities, erels, arg1, arg2, protnum)
        if sent_num==-1:
            removeFromRRels.append(rel)
        else:
            rrels[rel]['sent_num'] = sent_num

    for rel in removeFromRRels:
        del rrels[rel]

    return entities, erels, rrels

def arg2SentNum(rel, entities, erels, arg1, arg2, protnum):
    arg1sent = parseArg(entities, erels, arg1)
    arg2sent = parseArg(entities, erels, arg2)
    if arg1sent == arg2sent:
        return arg1sent
    else:
        #print('ERROR: arg2sent. arguments not from same sentence of relation ' + rel + ' in protocol ' + protnum)
        #print('not adding to set of relations')
        return -1

def parseArg(entities, erels, arg):
    if arg[0]=='E':
        arg = erels['actions'][arg]     # replace E relation arg with entity T
    if arg[0]=='T':
        return entities[arg]['sent_num']
    else:
        sys.exit('ERROR: ' + arg + 'is not a valid argument for a relation')

def parseRRels(entities, erels, line, protnum):
    lsplit = line.split()
    relJSON = {"id": 'm.' + protnum + '_' + lsplit[0].lower()}
    relJSON['relation_type'] = lsplit[1].lower()
    relJSON['arg1'] = lsplit[2].split(':')[1]
    relJSON['arg2'] = lsplit[3].split(':')[1]
    relJSON['sent_num'] = None

    return lsplit[0], relJSON     # return values in line, with id attached for easy access later

def parseERels(entities, erels, line, protnum):
    lsplit = line.split()
    # 2 parts
    #1. E#: alias for action = T#.
    action = lsplit[1].split(':')[1]
    erels['actions'][lsplit[0]] = action

    #2. for each relation following action, use action as arg1 and T# following relation as arg2
    for i in range(len(lsplit)-2):
        rel_type, ent = lsplit[i+2].split(':')
        rel_type = re.sub(r'\d+', '', rel_type)
        relJSON = {"id": 'm.' + protnum + '_' + lsplit[0].lower() + '_' + str(i)}
        relJSON['relation_type'] = rel_type.lower()
        relJSON['arg1'] = action
        relJSON['arg2'] = ent
        relJSON['sent_num'] = None
        erels['rels'][lsplit[0]+'_'+str(i)] = relJSON

def parseEntities(line, protnum, sentences):
    numsemicol = line.count(';')
    lsplit = line.split()
    entJSON = {'id': 'm.' + protnum + '_' + lsplit[0].lower()} # id mimics format of sub/obj_id in riedel_raw
    entJSON['entity_type'] = lsplit[1].lower()
    entJSON['start_index'] = int(lsplit[2])
    entJSON['end_index'] = int(lsplit[numsemicol+3])
    # combine rest of words in split to create token
    entJSON['token'] = "_".join(lsplit[numsemicol+4:])
    entJSON['sent_num'] = searchSentForToken(sentences, entJSON['start_index'], entJSON['end_index'])

    return lsplit[0], entJSON     # return values in line, with id attached for easy access later

def searchSentForToken(sentences, token_start, token_end):
    for sent in range(len(sentences)):
        if sentences[sent]['start_index']<= token_start and sentences[sent]['end_index'] > token_end:
            return sent
    return -1

######################### create training JSONS from sentence entities/relations
def createSentJSON(relations, training, sentences, entities, erels, rrels, protnum, nlp_wrapper, pp):
    for rel in erels['rels']:
        rtest = createRelSubObjJSON(erels['rels'][rel], sentences, entities, erels, rrels, nlp_wrapper)
        if rtest['rel'] in relations:
            relations[rtest['rel']] += 1
        else:
            relations[rtest['rel']] = 1
        training.append(rtest)

    for rel in rrels:
        rtest = createRelSubObjJSON(rrels[rel], sentences, entities, erels, rrels, nlp_wrapper)
        if rtest['rel'] in relations:
            relations[rtest['rel']] += 1
        else:
            relations[rtest['rel']] = 1
        training.append(rtest)

    return training

def createRelSubObjJSON(rel, sentences, entities, erels, rrels, nlp_wrapper):
    rtest = {}

    # add relation
    arg1 = rel['arg1'] if rel['arg1'][0] == 'T' else erels['actions'][rel['arg1']]
    arg2 = rel['arg2'] if rel['arg2'][0] == 'T' else erels['actions'][rel['arg2']]

    rtest["rel"] = '/' + entities[arg1]['entity_type'] + '/' + entities[arg2]['entity_type'] + '/' + rel['relation_type']

    # add subject id
    rtest["sub_id"] = entities[arg1]['id']

    # add subject
    rtest["sub"] = entities[arg1]['token'].lower()

    # add object
    rtest["obj"] = entities[arg2]['token'].lower()

    # add sentence
    rtest["sent"] = sentences[rel['sent_num']]['sent']

    # add rsent (lowercase sentence)
    rtest["rsent"] = rtest["sent"].lower()

    # add object id
    rtest["obj_id"] = entities[arg2]['id']

    # add openie
    rtest["openie"] = createOpenIEJSON(rtest['sent'], nlp_wrapper)

    # add corenlp
    rtest["corenlp"] = createOpenIEJSON(rtest['sent'], nlp_wrapper)


    return rtest

def createOpenIEJSON(text, nlp_wrapper):
    openie_json = nlp_wrapper.annotate(text, properties={'annotators': 'openie, depparse, tokenize', 'timeout': '50000', 'outputFormat': 'json'})
    del openie_json['sentences'][0]['enhancedDependencies']
    del openie_json['sentences'][0]['enhancedPlusPlusDependencies']
    return openie_json

def createCoreNLPJSON(text, nlp_wrapper):
    corenlp_json = nlp_wrapper.annotate(text, properties={'annotators': 'parse, depparse, entitymentions, kbp, tokenize', 'timeout': '50000', 'outputFormat': 'json'})
    del corenlp_json['sentences'][0]['enhancedDependencies']
    del corenlp_json['sentences'][0]['enhancedPlusPlusDependencies']
    return corenlp_json

def printTrainingToFile(training, outfilename, relationfilename, entitytypefile, relations, entities):
    outf = open(outfilename, 'w+')
    for rtest in training:
        outf.write(json.dumps(rtest))
    outf.close()

    relf = open(relationfilename, 'w+')
    relf.write(json.dumps(relations))
    relf.close()

    entf = open(entitytypefile, 'w+')
    entf.write(json.dumps(entities))
    entf.close()

########################### MAIN ####################

nlp_wrapper = StanfordCoreNLP('http://localhost:9000/')
path = 'protocols/train/'
train_files = [f for f in listdir(path) if isfile(join(path, f))]
pp = pprint.PrettyPrinter(indent=4)
training = []
outfilepath = 'protocols/wlp_raw/wlp_data/wlp_train.json'
relationfilepath = 'protocols/wlp_raw/wlp_relation2id.json'
entitytypefile = 'protocols/wlp_raw/type_info.json'
relations = {}
entitytype = {}

# open all text files in training and corresponding annotation files
i = 0
now = time.time()
for file in train_files:
    if file[-4:]=='.txt':
        protnum = file[file.find('_')+1:file.find('.')]
        txtfile = open(path+file, "r", encoding="utf8")
        annfile = open(path+file[:-4]+'.ann', "r", encoding="utf8")
        sentences, entities, erels, rrels = parseProtocol(txtfile, annfile, protnum)
        training = createSentJSON(relations, training, sentences, entities, erels, rrels, protnum, nlp_wrapper, pp)
        txtfile.close()
        annfile.close()
        i+=1
        if i%1==0:
            print(time.time()-now)
            break

# reformat entitytype
entitytype = {}
for k,v in entities.items():
    entitytype[entities[k]['id']] = ['/'+entities[k]['entity_type']]

print(len(training))
print(relations)
printTrainingToFile(training, outfilepath, relationfilepath, entitytypefile, relations, entitytype)
