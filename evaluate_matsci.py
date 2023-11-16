import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import os
from collections import defaultdict
import random
import Levenshtein

predefined_tasks = ['named entity recognition','slot filling','relation classification','event extraction','sentence classification','paragraph classification','synthesis action retrieval']
question_type_dict = {'named entity recognition':0,'paragraph classification':1,'relation classification':2,'event extraction':3,'synthesis action retrieval':4,'sentence classification':5,'slot filling':6}

def read_json(line):
    try:
        jdata = json.loads(line)
    except:
        jdata = dict()
    return jdata

def most_similar_answer(a,answer_set):
    a = a.strip().replace(' ', '')
    if(a in answer_set):
        return a
    dis = [Levenshtein.distance(a,x) for x in answer_set]
    idx = np.argmin(dis)
    return answer_set[idx]

def decoding(true,pred,qtype,res_dict):
    y_true = defaultdict(list)
    y_pred = defaultdict(list)
    
    for x,y,t in zip(true,pred,qtype):
        x = x.lower()
        y = y.lower()
        t = int(t)
        if (t==0):
            answer_map = res_dict['t_type_dict']
            answer_set = res_dict['t_type_set']
            y_true['ner'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['ner'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==1):
            answer_map = res_dict['pc_type_dict']
            answer_set = res_dict['pc_type_set']
            y_true['pc'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['pc'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==2):
            answer_map = res_dict['r_type_dict']
            answer_set = res_dict['r_type_set']
            y_true['re'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['re'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==3):
            x = x.strip().replace(' ', '')
            y = y.strip().replace(' ', '')
            if (len(x)==0 and len(y)==0):
                y_pred['arg'].append(1)
            elif (len(x)==0):
                y_pred['arg'].append(0)
            elif (len(y)==0):
                answer_map = res_dict['e_role_dict']
                answer_set = res_dict['e_role_set']
                tmp_x = x.split(',')
                for a in tmp_x:
                    true_role = a.split(':')[1]
                    y_pred['arg'].append(0)
                    y_true['ee'].append(answer_map[true_role.strip().replace(' ', '')])
                    y_pred['ee'].append(answer_map[most_similar_answer(' ',answer_set)])
            else:
                tmp_x = x.split(',')
                tmp_y = y.split(',')
                answer_map = res_dict['e_role_dict']
                answer_set = res_dict['e_role_set']
                if(len(tmp_x)==len(tmp_y)):
                    pass
                elif(len(tmp_x)<len(tmp_y)):
                    tmp_y = tmp_y[0:len(tmp_x)]
                else:
                    tmp_y = tmp_y + [':']*(len(tmp_x)-len(tmp_y))
                for a,b in zip(tmp_x,tmp_y):
                    try:
                        true_arg,true_role = a.split(':')
                        pred_arg,pred_role = b.split(':')
                        if (true_arg==pred_arg):
                            y_pred['arg'].append(1)
                        else:
                            y_pred['arg'].append(0)
                        y_true['ee'].append(answer_map[true_role.strip().replace(' ', '')])
                        y_pred['ee'].append(answer_map[most_similar_answer(pred_role,answer_set)])
                    except:
                        true_arg,true_role = a.split(':')
                        y_pred['arg'].append(0)
                        y_true['ee'].append(answer_map[true_role.strip().replace(' ', '')])
                        y_pred['ee'].append(answer_map[most_similar_answer(' ',answer_set)])
        if (t==4):
            answer_map = res_dict['sar_dict']
            answer_set = res_dict['sar_set']
            y_true['sar'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['sar'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==5):
            answer_map = res_dict['sc_type_dict']
            answer_set = res_dict['sc_type_set']
            y_true['sc'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['sc'].append(answer_map[most_similar_answer(y,answer_set)])
        if (t==6):
            answer_map = res_dict['sf_type_dict']
            answer_set = res_dict['sf_type_set']
            y_true['sf'].append(answer_map[x.strip().replace(' ', '')])
            y_pred['sf'].append(answer_map[most_similar_answer(y,answer_set)])
    return y_true,y_pred

def metric(labels,preds):
    assert len(labels)==len(preds)
    if (len(labels)==0):
        return 0,0
    micro_f1 = f1_score(labels,preds,average='micro')
    macro_f1 = f1_score(labels,preds,average='macro')
    return micro_f1,macro_f1

if __name__=='__main__':    
    res_dict = json.load(open('res_dict.json','r'))
    f = open('prediction_outputs.txt','r') # The outputs of LLM for MATSCI-BENCHMARK
    data = json.load(f)
    
    y_true = defaultdict(list)
    y_pred = defaultdict(list)
    for sample in data:
        if (len(sample)==0):
            continue
        for task in predefined_tasks:
            real_ans = []
            pred_ans = []
            qtype = []
            if (task in sample['instruction']):
                true = sample['output'].lower()
                pred = sample['generate'].lower()
                qtype.append(question_type_dict[task])
                real_ans.append(true)
                pred_ans.append(pred)
            decode_true,decode_pred = decoding(real_ans,pred_ans,qtype,res_dict)
            for key in decode_true:
                y_true[key] += decode_true[key]
            for key in decode_pred:
                y_pred[key] += decode_pred[key]
                
    for key in y_true:
        micro,macro = metric(y_true[key],y_pred[key])
        print('task = {} micro-f1 = {} macro-f1 = {}'.format(key,micro,macro))
