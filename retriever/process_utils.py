import os
from tqdm import tqdm
import json
import re
import string
from sentence_transformers import SentenceTransformer, util
from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np

# embedder = SentenceTransformer('paraphrase-distilroberta-base-v2')
# tagger = SequenceTagger.load('ner')
def re_trans(text, ignore=False):
    text = re.sub(r"([.^$*+\\\[\]])", r"\\\1", text)
    text = fr"\W{text}\W"
    
    if ignore:
        return re.compile(text, re.IGNORECASE)
    else:
        return re.compile(text)
    
    
def re_sub_trans(before, after, text):
    before = re.sub(r"([.^$*+\\\[\]])", r"\\\1", before)
    before = fr"(\W)({before})(\W)"
    try:
        r =  re.compile(before)
    except:
        print(before, after, text)
    # print(r)
    
    if re.match('\d', after):
        pass
        # with open("shuzi.txt", 'a') as f:
        #     f.write(before+'\t'+after+'\t'+text+'\n')
        
        # print("before", before)
        # print("after", after)
        # print(text)
        # after = fr"\1%{after}\3"
        # text = re.sub(r, after, text)
        # text = re.sub("%", "", text)
    else:
        after = fr"\1{after}\3"
        text = re.sub(r, after, text)
        
    
    return text

    
def match_one_entity(entity, text, embedder, tagger):
    results = []
    for m in re.finditer(re_trans(entity), text): # Check whether the entity name matches the text
        results.append([m.group()[1:-1], m.start()+1, m.end()-1]) 
        # print('first')
    if results:
        return results, []
    
    for m in re.finditer(re_trans(entity, ignore=True), text):
        results.append([m.group()[1:-1], m.start()+1, m.end()-1])
    if results:
        return results, []
    
    buchong = re.search("\s*\(.*?\)\s*",entity) 
    # processing entities with brackets
    if buchong:
        buchong = re.sub("\s*\(", "",buchong.group()) 
        buchong = re.sub("\)\s*", "",buchong) 
        entity2 = re.sub("\s*\(.*?\)\s*", "", entity).strip() 
        results = kuohao(entity2, buchong, text)
        # entity = "Albert Einstein (physicist)" → entity2 = "Albert Einstein", buchong = "physicist"
        if results:
            return results, []

    # If no matching entity is found in the text, perform NER using Flair
    replace_list = use_NER(entity, text, embedder, tagger) 
    return [], replace_list

def kuohao(entity, buchong, text):
    results = []
    # '''['"]*'''+entity+'''['"]*'''
    for m in re.finditer(re_trans(entity), text): 
        results.append([m.group()[1:-1], m.start()+1, m.end()-1]) 
    if not results:
        for m in re.finditer(re_trans(entity, ignore=True), text):
            results.append([m.group()[1:-1], m.start()+1, m.end()-1])
            
    if results:
        results_buchong = []
        for m in re.finditer(re_trans(buchong, ignore=True), text):
            results_buchong.append([m.group()[1:-1], m.start()+1, m.end()-1])
            
        return merge_kuohao(results, results_buchong, text)
    
    return results




def merge_kuohao(results, results_buchong, text):
    # print('merge_kuohao')
    if not results_buchong:
        return results
    # print(results, results_buchong)
    results2 = []
    for r in results:
        flag = False
        for rb in results_buchong:
            if -1<rb[1]-r[2]<4:
                flag = True
                r2 = [text[r[1]:rb[2]], r[1], rb[2]]
                results2.append(r2)
                break
            elif -1<r[1]-rb[2]<4:
                flag = True
                r2 = [text[rb[1]:r[2]], rb[1], r[2]]
                results2.append(r2)
                break
                
        if not flag:
            results2.append(r)
                
    return results2
                
            
# Match each entity with the text, if necessary, use the use_NER function to replace the text and re-matching.       
def match_entities(entities, text, embedder, tagger, if_replace=True):
    all_replace_list = [] 
    all_results = {} # entity names as keys, entity information (entity name, start position, end position) as values.
    for i, entity in enumerate(entities):
        results, replace_list = match_one_entity(entity, text, embedder, tagger)
        if results:
            all_results.update({entity: results})
        all_replace_list += replace_list
            
    all_replace_list = set(all_replace_list)
    if if_replace and all_replace_list: # re-matching
        for replace_list in all_replace_list:
            # print(replace_list)
            text = re_sub_trans(replace_list[0], replace_list[1], text) # Replace the entity name with the entity name obtained by NER

        all_results = {}
        for i, entity in enumerate(entities): 
            results, _ = match_one_entity(entity, text, embedder, tagger)
            if results:
                all_results.update({entity: results})
                
        return text, all_results
    else: # If no entity is replaced, return the original text and the entity information obtained by matching.
        return text, all_results
        





# Process the text with NER, and use the specified entity as a replacement candidate if the embedding similarity is high
SIMILARITY = 0.75
def use_NER(entity, text, embedder, tagger):
    entity_embed = embedder.encode(entity)
    entity_len = len(entity.split())
    
    replace_list = []
    for turn in re.split("<.*?>", text):
        if turn.strip():
            sentence = Sentence(turn)
            tagger.predict(sentence)
            for entity_text in sentence.get_spans('ner'):
                score = util.pytorch_cos_sim(entity_embed, embedder.encode(entity_text.text))
                # print("score", score, entity_text.text)
                if score > SIMILARITY:#match
                    # print("similar")
                    # print(entity_len, len(entity_text.text.split()))
                    # if abs(len(entity_text.text.split())-entity_len)/entity_len<0.5:
                    replace_list.append((entity_text.text, entity))
    return replace_list


def entity_linking(kg_nodes, kg_embeddings, reponse, embedder, tagger):
    matched_entities = []
    sentence= Sentence(reponse.strip())
    tagger.predict(sentence)
    
    for entity_text in sentence.get_spans('ner'):
        entity_candidate = entity_text.text 
        entity_candidate_embed = embedder.encode(entity_candidate)  
        
        
        potential_nodes = [node for node in kg_nodes if entity_candidate.lower() in node.lower()]
        
        for node in potential_nodes:
            score = util.pytorch_cos_sim(entity_candidate_embed, kg_embeddings[node])
            if score > SIMILARITY:  
                matched_entities.append((entity_candidate, node))
    return matched_entities


# Tokenize the text and create an index corresponding to the token positions of each entity    
def get_tokenized_idx(text, all_results, name_to_id, tokenizer, mod):
    original = tokenizer.encode(text)[1:-1] # remove [CLS] and [SEP] at the beginning and end
    # print(original, len(original))

    all_results = sorted(all_results.items(), key=lambda items: len(tokenizer.encode(items[0])), reverse=True)
    
    history = []
    all_idxs = {}
    for entity, results in all_results:
        if entity == '':
            continue 
        entity_id = name_to_id.get(entity)
        if not entity_id:
            continue
        if isinstance(entity_id, set):  
            entity_id = next(iter(entity_id))  
        # print(entity, entity_id)
        total = [0]*len(original)
        # print("results:", results)
        for res in results:
            s = res[1] # start position
            e = res[2] # end position
            
            # s2: the "exact" start position of the entity considering spaces
            if text[s-1] == " ": # If the character before the start position is a space
                s2 = s-1
            elif text[s-2] == " ":
                s2 = s-2
            elif text[s-3] == " ":
                s2 = s-3
            else:
                s2 = s
                
            # e2: the "exact" end position of the entity considering spaces and "<"   
            if text[e] in [" ", "<"]:
                e2 = e
            elif len(tokenizer.encode(text[s2:e]))==len(tokenizer.encode(text[s2:e+1])):
                e2 = e+1
            else:
                e2 = e
            
            # check whether the start and end positions of the entity overlap with those of other entities.
            if justify_overlap(s2, e2, history):
                continue
                
            history.append((s2, e2))
            front = tokenizer.encode(text[:s2])[1:-1] # part before the entity
            entity_idxs = tokenizer.encode(text[s2:e2])[1:-1] # part of the entity
            behind = tokenizer.encode(text[e2:])[1:-1] # part after the entity
                
                
            if original!= front+entity_idxs+behind:
                print(text)
                print(all_results)
                print("original", original)
                print("front", front)
                print("entity_idxs", entity_idxs)
                print("behind", behind)
                print("res", res)
                print("s2, e2", s2, e2)
                
                
            assert original == front+entity_idxs+behind
            
            if mod=='all': # Assign an ID to all token positions of the entity
                total[len(front):len(front)+len(entity_idxs)] = [entity_id]*len(entity_idxs)
            elif mod=='first': # Assign an ID to the first token position of the entity
                total[len(front)] = entity_id
            else:
                assert False
            # print("total", total)
        all_idxs.update({entity: total})
        
    print("Token positions of the entity", all_idxs)
            
    return all_idxs

    
def justify_overlap(s, e, history):
    assert e > s
    for s1, e1 in history:
        if not (e1<s or s1>e):
            return True
    return False    

remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def filter_str(sentence):
    sentence = re.sub(remove_nota, '', sentence)
    sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()
    
    
def only_english(s):
    s = filter_str(s)
    result = []
    s = re.sub('[0-9]', '', s).strip()
    if not s:
        return True
    
    other_words = re.compile(u"[\u4e00-\u9fa5\uac00-\ud7ff\u30a0-\u30ff\u3040-\u309f]+")
    if re.findall(other_words, s):
        return False
    
    # unicode english
    # res = re.sub('[a-zA-Z]', '', s).strip()
    re_words = re.compile(u"[a-zA-Z]")
    res = re.findall(re_words, s)  
    if res:
        return True
    else:
        return False
    
    
class Triple_id():
    def __init__(self, file_entity_id, file_relation_id):
        with open(file_entity_id, "r", encoding='utf-8') as f_entity, \
        open(file_relation_id, "r", encoding='utf-8') as f_relation:
            entity_id = {} # entities to ids mapping
            id_entity = {} # ids to entities mapping
            add_tokens = ["<pad>"]
            for i, token in enumerate(add_tokens):
                entity_id[token] = i
                id_entity[i] = token

            entity_line = f_entity.readlines()
            for i, entityid in enumerate(entity_line):
                if i == 0: 
                    num_entity = int(entityid.strip())+len(add_tokens) 
                    continue
                items = entityid.strip().split('\t')
                assert len(items) == 2, f"無効なフォーマット: 行 {i+1} -> {entity_line.strip()}"
                entity_name = items[0]
                entity_id_num = int(items[1]) + len(add_tokens) # considering the additional tokens
                id_entity[entity_id_num] = entity_name 
                """
                items = entityid.split('\t')
                assert len(items) == 2
                assert items[0] not in entity_id.keys() 
                entity_id[items[0]] = int(items[1])+len(add_tokens)
                id_entity[int(items[1])+len(add_tokens)] = items[0]
                """
                
                if entity_name in entity_id:
                    entity_id[entity_name].append(entity_id_num)
                else:
                    entity_id[entity_name] = [entity_id_num]
                
            assert num_entity == len(id_entity), f"Mismatch in the number of entities: {num_entity} != {len(id_entity)}"
            #assert num_entity == len(id_entity.keys())
            self.num_entity = num_entity
            self.entity_id = entity_id
            self.id_entity = id_entity

            relation_id = {}
            id_relation = {}
            relation_line = f_relation.readlines()
            for i, relationid in enumerate(relation_line):
                if i == 0: 
                    num_relation = int(relationid.strip())
                    continue
                items = relationid.strip().split('\t')
                assert len(items) == 2, f"無効なフォーマット: 行 {i+1} -> {relation_line.strip()}"
                assert items[0] not in relation_id.keys()
                relation_id[items[0]] = int(items[1])
                id_relation[int(items[1])] = items[0]
                
            assert num_relation == len(id_relation.keys()), f"Mismatch in the number of relations: {num_relation} != {len(id_relation)}"
            self.id_relation = id_relation
            self.relation_id = relation_id
            
            #Cannot merge because the relation and entity overlap


    def get_triple_id(self, word, relation=True):
        if relation:
            return self.relation_id[word.lower()] + self.num_entity
        else: # in the case of entity
            ids = self.entity_id.get(word.lower())
            if ids is None:
                raise ValueError(f"Cannot find entity '{word}'")
            return ids
    
    def get_triple_name(self, id, relation=True):
        if relation:
            relation_id = id - self.num_entity
            name = self.id_relation.get(relation_id)
            if name is None:
                raise ValueError(f"The relation name corresponding to relation ID '{relation_id}' was not found.")
            assert isinstance(name, str), f"Multiple relation names exist for relation ID '{id}'"
            return name
        else:    
            name = self.id_entity.get(id)
            if name is None:
                raise ValueError(f"The entity name corresponding to entity ID '{id}' was not found.")
            assert isinstance(name, str), f"Multiple entity names exist for entity ID '{id}'"
            return name
    
            
"""  
def create_text_id(hist_response_ids, hist, response, tokenizer):
    if not hist_response_ids:
        hist_ids = tokenizer.encode(hist)[1:-1]
        response_ids = tokenizer.encode(response)[1:-1] ##
        len_hist_response_ids = len(hist_ids)+1+len(response_ids) ##
        #return [0]*len(hist_ids)
        return [0]*len_hist_response_ids
    
    all_entityids = []
    for entity, entityids in hist_response_ids.items():
        all_entityids.append(np.array(entityids))
    hist_response_ids = list(np.sum(all_entityids, axis = 0))
    
    hist_ids = tokenizer.encode(hist)[1:-1]
    response_ids = tokenizer.encode(response)[1:-1]
    
    # print(len(hist_response_ids), len(hist_ids), len(response_ids))
    assert len(hist_response_ids) == len(hist_ids)+1+len(response_ids)
    #return hist_response_ids[:len(hist_ids)]
    return hist_response_ids
"""  

def create_text_id(text_ids, text, tokenizer):
    if not text_ids:
        text_ids = tokenizer.encode(text)[1:-1]
        len_text_ids = len(text_ids)
        return [0]*len_text_ids
    
    all_entityids = []
    for entity, entityids in text_ids.items(): # e.g. {'valparaiso university': [0, 0, 0, 0, 0, 0, 2156, 2156, 2156, 2156, 2156, 0]}
        all_entityids.append(np.array(entityids))
    text_ids = list(np.sum(all_entityids, axis = 0))
    
    return text_ids
    