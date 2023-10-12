import json
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some llm info.')
    parser.add_argument('--path', type=str, default="none")
    args = parser.parse_args()
    claude_eval_res_path = args.path
    
    with open(claude_eval_res_path,'r') as f1:
        eval_res_list = json.load(f1)
    
    selected_instructions = []
    for eval_res in eval_res_list:
        accuracy = eval_res['Accuracy']
        relevance = eval_res['Relevance']
        completeness = eval_res['Completeness']
        reasonableness = eval_res['Reasonableness']
        avg_score = (accuracy + relevance + completeness + reasonableness)/4.0
        if (avg_score>=95 and accuracy>=90 and relevance>=90 and completeness>=90 and reasonableness>=90):
            instruction = {}
            instruction['input'] = eval_res['input']
            instruction['output'] = eval_res['output_text']
            instruction['instruction'] = eval_res['instruction']
            selected_instructions.append(instruction)