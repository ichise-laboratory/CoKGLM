import os
import json
import torch
import argparse
import sys
import re
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import csv
import math
import json
from argparse import ArgumentParser
import ast
import random
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
from flair.data import Sentence
from flair.models import SequenceTagger
from process_utils import *
from myutils import *
import spacy
from fuzzywuzzy import fuzz


def build_dataset(tripleid, in_path, out_path, mod, Graph, name2entity, kg_triple_id, entity2name, relation2name, filter_no_english=False):
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_tokens(['<sep>','<triple>','<user>','<assistant>','<response>'])
    embedder = SentenceTransformer('paraphrase-distilroberta-base-v2')
    tagger = SequenceTagger.load('ner')
    nlp = spacy.load("en_core_web_sm")
    
    with open(in_path, 'r') as f_in, open(out_path,"w") as fout:
        reader = csv.DictReader(f_in, delimiter=',')
        #next(reader)
        writer = csv.writer(fout)
        writer.writerow(['entity_relation_ids', 'memory_bank', 'entity_triple_counts', 'history_response', 'corrupt_entity', 'original_entity', 'hallucination_tags', "data_id"])
        pre = 'Given the knowledge:'
        text_prefix = 'Text to check:'
        pre_len = len(tokenizer.tokenize(pre))
        text_prefix_len = len(tokenizer.tokenize(text_prefix))
        
        original_line = 0
        hallucination_line = 0
        
        for idx, l_in in tqdm(enumerate(reader)):
            print("idx", idx)
            data_id = l_in["id"]
            corrupt_response = l_in["declarative"]
            corrupt_entity_str = l_in["gen_ans"]
            original_entity_str = l_in["gt_ans"]
            corrupt_entity = parse_to_list(corrupt_entity_str)
            original_entity = parse_to_list(original_entity_str)
               
            corrupt_entity_lower = [item.lower() if isinstance(item, str) else item for item in corrupt_entity]
            original_entity_lower = [item.lower() if isinstance(item, str) else item for item in original_entity]
            
            hallucination_tags = ast.literal_eval(l_in["hallucinated"])
            if hallucination_tags == 1:
                hallucination_line += 1
            elif hallucination_tags == 0:
                original_line += 1
                    
            # perform NER using spaCy
            doc = corrupt_response.lower()
            doc = nlp(doc)
            matched_entities = [ent.text for ent in doc.ents]
            matched_entities = list(dict.fromkeys(matched_entities))
            print("matched_entities:", matched_entities)
            doc_corrupt_response_vector = doc.vector                
            
            entity_triple_counts = {}  # the number of triples surrounding each entity  

            MAX_TRIPLES_PER_ENTITY = 5  # the number of triples to retrieve for each entity
           
            for entity in matched_entities:
                
                lower_entity = entity.lower()
                triple_list=[]
                count = 0
                triple_count = 0
                        
                for node in Graph.nodes():
                    
                    # If at least one triple is retrieved for an entity, proceed to the next entity.
                    if len(triple_list)> 0:
                        break  
                    
                    # perform entity linking
                    if fuzz.ratio(lower_entity, node.lower()) >= 99:
                        # identifying a list of candidate entities through string matching and filtering them based on the similarity between all surrounding triples and the context.
                        lower_entity_ids = name2entity.get(node.lower()) 
                        print("candidate_entity_ids:", lower_entity_ids)
                        
                        best_id = None,
                        best_score = -1
                        for lower_entity_id in lower_entity_ids:
                            triple_text_vectors = []
                            extracted_triples, id_list = extract_triple(kg_triple_id, lower_entity_id, entity2name, relation2name)
                            
                            for u, key, v in extracted_triples:
                                triple_text = f"{u} {key} {v}"
                                triple_text_nlp = nlp(triple_text)
                                triple_text_vectors.append(triple_text_nlp.vector)
                            
                            # If no triples surrounding the entity, proceed to the next candidate entity.   
                            if len(triple_text_vectors) == 0: 
                                continue
                            
                           
                            avg_triple_vector = np.mean(triple_text_vectors, axis=0)
                            similarity = cosine_similarity([avg_triple_vector], [doc_corrupt_response_vector])[0][0]
                            
                            if similarity > best_score:
                                best_score = similarity
                                best_id = lower_entity_id
                        
                        print("best_id:", best_id)
                        
                        # Extract the surrounding triples from the linked entity.
                        extract_triples, extract_triples_id = extract_triple(kg_triple_id, best_id, entity2name, relation2name)
                
                        triple_count += len(extract_triples)
                        entity_triple_counts[int(best_id) + 1] = triple_count # <pad> (assigned an ID of 0) will later be added to entity2id
                        
                        # shuffle the extracted triples
                        combined_triples = list(zip(extract_triples, extract_triples_id))
                        random.shuffle(combined_triples)
                        extract_triples, extract_triples_id = zip(*combined_triples)
                        
                        # Sort the triples based on their similarity to the context.
                        similarity_scores = []
                        for i, ((u, key, v), (u_id, key_id, v_id)) in enumerate(zip(extract_triples, extract_triples_id)):
                            
                            # ensure the linked entity appears at the beginning of each triple
                            if int(u_id) != int(best_id):
                                u_id, v_id = v_id, u_id 
                                u, v = v, u
                                
                            triple = [u, key, v]
                            triple_ids = [u_id, key_id, v_id]
                            triple_text = f"{u} {key} {v}"
                            triple_text = nlp(triple_text)
                            triple_text_vector = triple_text.vector
                            similarity = cosine_similarity([triple_text_vector], [doc_corrupt_response_vector])[0][0]
                            similarity_scores.append((similarity, triple, triple_ids))

                        similarity_scores.sort(reverse=True, key=lambda x: x[0])
                        
                        for sim, triple, triple_ids in similarity_scores:
                            if count < MAX_TRIPLES_PER_ENTITY:
                                triple_list.append((triple, triple_ids))
                                count += 1
                                if len(triple_list) >= MAX_TRIPLES_PER_ENTITY:
                                    break
                        if count > 0:
                            break 
                                
                entity_relation_ids = [0]*pre_len
                memory_bank = []
                triples = []
                
                if_only_english = True
                all_entities = set()
                
                # Add the retrieved triples to entity_relation_ids
                name_to_id = {}
                for triple, triple_ids in triple_list:
                    sub, rel, obj = triple
                    sub_id, rel_id, obj_id = triple_ids
                    all_entities.add(sub)
                    all_entities.add(obj)
                            
                    sub_id = int(sub_id) + 1
                    obj_id = int(obj_id) + 1
                    #rel_id = int(rel_id) + tripleid.num_entity
                    rel_id = int(rel_id) + 17755 # +num_entity
                    rel = re.sub("[-_]",' ',rel)
                    
                    # filter the duplicate entity-to-ID mappings in name_to_id
                    # ensure all entities in triple_list have unique and consistent mappings to their respective ID
                    if sub in name_to_id and name_to_id[sub] != sub_id:
                        raise ValueError(f"Entity '{sub}' has been assigned a different ID: {name_to_id[sub]} vs {sub_id}")
                    name_to_id[sub] = sub_id
                    
                    if obj in name_to_id and name_to_id[obj] != obj_id:
                        print(f"Entity '{obj}' has been assigned a different ID: {name_to_id[obj]} vs {obj_id}")
                        continue  # Skip current triple since conflicting IDs were found for 'obj'
                    name_to_id[obj] = obj_id
                    
                    #if rel in name_to_id and name_to_id[rel] != rel_id:
                        #raise ValueError(f"リレーション '{rel}' に異なるIDが割り当てられている: {name_to_id[rel]} vs {rel_id}")
                    #name_to_id[rel] = rel_id
                    
                    memory_bank.append([sub_id, rel_id, obj_id])
                    
                    sub_len = len(tokenizer.tokenize(' '+sub))
                    rel_len = len(tokenizer.tokenize(' '+rel))
                    obj_len = len(tokenizer.tokenize(' '+obj))
                    if mod == 'all':
                        entity_relation_ids += [sub_id]*sub_len+[0]+[rel_id]*rel_len+[0]+[obj_id]*obj_len+[0] 
                    elif mod=='first':
                        entity_relation_ids += [sub_id]+[0]*sub_len+[rel_id]+[0]*rel_len+[obj_id]+[0]*obj_len
                    else:
                        assert False
                    
                    if filter_no_english:
                        if_only_english = if_only_english and only_english(sub) and only_english(rel) and only_english(obj)
                                
                    triple2 = sub+'<sep> '+rel+'<sep> '+obj
                    triples.append(triple2)
                    
                entity_relation_ids = entity_relation_ids[:-1] 
                
                input_text = pre+' '+'<triple> '.join(triples)+' '+text_prefix
                
                if not if_only_english:
                    print('no only', idx, input_text)
                    continue                  
                
                corrupt_response, all_results = match_entities(all_entities, corrupt_response, embedder, tagger, if_replace=True)
                corrupt_response_entity_relation_ids = get_tokenized_idx(corrupt_response, all_results, name_to_id, tokenizer, mod)
                text_entity_relation_ids = create_text_id(corrupt_response_entity_relation_ids, corrupt_response, tokenizer)
                                    
                input_text += corrupt_response 
                
                entity_relation_ids += [0]*text_prefix_len + text_entity_relation_ids
                entity_relation_ids = [0]+entity_relation_ids+[0] # <s> and </s>
                
                if len(entity_relation_ids) != len(tokenizer.encode(input_text)): # the mismatch indicates that no triples were retrieved for the entity, skip the current example
                    print(len(entity_relation_ids), len(tokenizer.encode(input_text)))
                    print("error entity_relation_ids", entity_relation_ids)
                    print("error input_ids", tokenizer.tokenize(input_text))
                    print("error input_text", input_text)
                    pass 
                
                else:
                    #print("final entity_relation_ids", entity_relation_ids)
                    #print("final input_text", input_text)
                    #print("final input_ids", tokenizer.tokenize(input_text)) 
                    writer.writerow([entity_relation_ids,
                                    memory_bank,
                                    entity_triple_counts,
                                    input_text,
                                    corrupt_entity,
                                    original_entity,
                                    hallucination_tags,
                                    data_id])

def parse_to_list(input_data):
    if isinstance(input_data, list):
        return input_data
    elif input_data:
        try:
            return ast.literal_eval(input_data) if isinstance(input_data, str) else [input_data]
        except Exception:
            return [input_data]
    return []           

def check_entity_id(path):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, row in enumerate(reader):
            entity_ids = set(ast.literal_eval(row[0]))
            
            entity_ids2 = set([0])
            for memory_bank in ast.literal_eval(row[1]):
                entity_ids2 |= set(memory_bank)
                
            if entity_ids != entity_ids2:
                print(i)
                print(entity_ids)
                print(entity_ids2)
                
            assert entity_ids == entity_ids2

def map_id_to_name(file_path):
    id2name = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            name, id = line.strip().split("\t")
            id2name[int(id)] = name
    return id2name
def map_name_to_id(file_path):
    name2id = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            name, id = line.strip().split("\t")
            if name not in name2id:
                name2id[name] = []
            name2id[name].append(id)
    return name2id

# Extract triples surrounding the entity
def extract_triple(kg_triple_id, entity_id, entity2name, relation2name):
    extracted_triples = []
    extracted_triples_id = []
    
    entity_id = int(entity_id)
    with open (kg_triple_id, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            head_id, tail_id, relation_id = map(int, line.strip().split("\t"))
            
            if head_id == entity_id or tail_id == entity_id:
                head_name = entity2name.get(head_id)
                tail_name = entity2name.get(tail_id)
                relation_name = relation2name.get(relation_id)
                
                extracted_triples.append((head_name, relation_name, tail_name))
                extracted_triples_id.append((head_id, relation_id, tail_id))
                #print("extracted_triples:", extracted_triples)
                #print("extracted_triples_id:", extracted_triples_id)
    
    return extracted_triples, extracted_triples_id
            
        
def main():
    parser = ArgumentParser()
    parser.add_argument("--file_entity_id", type=str, required=True, help="Path to entity_id")
    parser.add_argument("--file_relation_id", type=str, required=True, help="Path to relation_id")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output csv file")
    args = parser.parse_args()
    tripleid = Triple_id(args.file_entity_id, args.file_relation_id)
    
    # Load the KG
    freebase_file = "../kg/kg_triple.txt"
    kge_dir = "../kg"
    ext_entities_file = "../kg/ext_entities.txt"  
    annotator = Annotator(freebase_file=freebase_file,
                            kge_dir=kge_dir,
                            ext_entities_file=ext_entities_file)
    Graph = annotator.load_kg(freebase_file)
    entity2name = map_id_to_name(args.file_entity_id)
    name2entity = map_name_to_id(args.file_entity_id)
    relation2name = map_id_to_name(args.file_relation_id)
    kg_triple_id = "../kg/kg_triple_id.txt"
    
    build_dataset(tripleid=tripleid,
                  in_path=args.input_file,
                  out_path=args.out_file, 
                  mod="all",
                  Graph=Graph,
                  name2entity=name2entity,
                  kg_triple_id=kg_triple_id,
                  entity2name=entity2name,
                  relation2name=relation2name)
    check_entity_id(args.out_file)

"""
CUDA_VISIBLE_DEVICES=0 python build_dataset.py \
--input_file qa2hall_data.csv \
--out_file ../QA2HALL/retrieved_data/extract5_per_entity_new.csv \
--file_entity_id ../kg/entity2id.txt \
--file_relation_id ../kg/relation2id.txt
"""
   
if __name__ == "__main__":
    main()

