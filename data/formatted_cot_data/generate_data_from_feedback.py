import openai
import threading
from multiprocessing import Pool
import time
import random
import json
openai.api_key = "xxxxxxxxxxxxx"

thread_num=4

Ability = {
    'accuracy': "The accuracy of the given text (including <input>, <instruction> and <output>) is evaluated by comparing the <output> with known facts or credible sources. This involves checking the accuracy of any claims or statements made in the <output>, and verifying that they are supported by evidence.",
    'completeness': "The completeness of the given text (including <input>, <instruction> and <output>) is evaluated by assessing how fully the <output> addresses the <instruction>, including all sub-questions. Consider both depth and conciseness.",
    'reasonableness': "The reasonableness of the given text (including <input>, <instruction> and <output>) is evaluated by considering how logically consistent the <output> is, with no obvious contradictions."
    }

def return_prompt(data,score):
    input_text = data['input']
    output_text = data['output']
    instruction = data['instruction']
    accuracy = score['Accuracy']
    completeness = score['Completeness']
    reasonableness = score['Reasonableness']
    topic_list = ['Nanomaterials', 'Polymers', 'Composites', 'Biomaterials', 'Metals', 'Semiconductors', 'Superconductors', 'Ceramics', 'Glass', 'Smart materials', 'Optical materials', 'Magnetic materials', 'Graphene', 'Carbon nanotubes', 'Energy materials', 'Construction materials', 'Electronic materials', 'Thermoelectric materials', 'Bio-inspired materials', 'Self-healing materials']
    task_list = [
        "Open-ended generation",
        "Classification",
        "Named Entity Recognition",
        "Question answering",
        "Editing",
        "Summarization",
        "Writing",
        "Analysis",
        "Code interpretation",
        "Commonsense reasoning",
        "Information Extraction",
        "Clustering",
        "Topic modeling",
        "Sentiment analysis",
        "Grammar correction",
        "Machine reading comprehension",
        "Event Extraction",
        "Text simplification",
        "Part-of-speech tagging",
        "Relation extraction"
    ]
    system_prompts = []
    for i,metric in enumerate([accuracy,completeness,reasonableness]):
        ability = 'accuracy'
        if (i==0 and metric<100):
            ability = 'accuracy'
        elif(i==1 and metric<100):
            ability = 'completeness'
        elif(i==2 and metric<100):
            ability = 'reasonableness'
        else:
            continue
        desp = Ability[ability]
        if (len(input_text)>0):
            system_prompt = "You need to provide diverse task instructions and corresponding responses as much as possible based on the given text for finetuning LLAMA model. Note its format is latex and you should process it properly. Requirements:\n"
            system_prompt += "1. The given text is: " + input_text + ".\n"
            system_prompt += "2. The LLAMA model is currently not performing well on the following data sample:\n <input>: {}\n <instruction>: {}\n <output>: {}\n You should analyze insufficent points of the given data sample and then generate more targeted task instructions and corresponding responses to help LLAMA model improve its insufficient points.\n Specifically, the instruction data should focus on improving the LLAMA model's ability to {}.\n {}\n".format(input_text, instruction, output_text, ability, desp)
            # other requirements
            system_prompt += "3. If encountering instructions that cannot be processed (cannot be answered solely based on the text), provide a response indicating that it cannot be processed.\n"
            system_prompt += "4. Unless specifically required, please use English. Instructions can be command sentences, questions, or other appropriate types.\n"
            system_prompt += "5. Generate an appropriate and realistic <instruction>, which should not only contain simple placeholders. <instruction> should provide substantive content, and be challenging. The number of words should not exceed " + str(random.randint(100, 1000)) + ".\n"
            system_prompt += "6. <output> should be an appropriate and realistic response to the instruction, and cannot simply reply to the request with acceptance or refusal. If additional information is needed to respond, please try to predict the user's intention and attempt to reply. The content of <output> should be less than " + str(random.randint(100, 1000)) + " words.\n\n"
            system_prompt += "Please provide 5 JSON format data that meet the requirements. The json should only contain the following fields: instruction, and output. The JSON format data should not be numbered, and each data should be on a separate line. There should be no spaces between each line.\n"
        else:
            system_prompt = "You need to provide diverse task instructions and corresponding responses as much as possible for finetuning LLAMA model. Requirements:\n"
            system_prompt += "1. Cover the following topics: " + "、".join(random.sample(topic_list, 5)) + ".\n" + "Diverse types of instructions, such as: " + "、".join(random.sample(task_list, 5)) + ", etc.\n"
            system_prompt += "2. The LLAMA model is currently not performing well on the following data sample:\n <instruction>: {}\n <output>: {}\n You should analyze insufficent points of the given data sample and then generate more targeted task instructions and corresponding responses to help LLAMA model improve its insufficient points.\n Specifically, the instruction data should focus on improving the LLAMA model's ability to {}.\n {}\n".format(instruction, output_text, ability, desp)
            # other requirements
            system_prompt += "3. If encountering instructions that cannot be processed (cannot be answered solely based on the text), provide a response indicating that it cannot be processed.\n"
            system_prompt += "4. Unless specifically required, please use English. Instructions can be command sentences, questions, or other appropriate types.\n"
            system_prompt += "5. Generate an appropriate and realistic <instruction>, which should not only contain simple placeholders. <instruction> should provide substantive content, and be challenging. The number of words should not exceed " + str(random.randint(100, 1000)) + ".\n"
            system_prompt += "6. <output> should be an appropriate and realistic response to the instruction, and cannot simply reply to the request with acceptance or refusal. If additional information is needed to respond, please try to predict the user's intention and attempt to reply. The content of <output> should be less than " + str(random.randint(100, 1000)) + " words.\n\n"
            system_prompt += "Please provide 5 JSON format data that meet the requirements. The json should only contain the following fields: instruction, and output. The JSON format data should not be numbered, and each data should be on a separate line. There should be no spaces between each line.\n"
        system_prompts.append(system_prompt)
    return system_prompts

def generate_response(data,score):
    prompts = return_prompt(data,score)
    result = []
    for prompt in prompts:
        retry_time = 2
        while (retry_time>0):
            try:
                time.sleep(0.5*(3-retry_time))
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",    # here we use `gpt-3.5-turbo` model, while Stanford-Alpaca uses `text-davinci-003`
                    stop=None,              # The stopping sequence for the generated response, if any (not used here)
                    temperature=0.0,        # The "creativity" of the generated response (higher temperature = more creative)
                    messages=[
                      {"role": "user", "content": prompt},
                    ]
                    )
                response = response["choices"][0]["message"]["content"]
                for line in response.split('\n'):
                    jdata = json.loads(line)
                    jdata['input'] = data['input']
                    result.append(jdata)
                retry_time = 0
            except:
                retry_time -= 1
                print('current retry_time = ',retry_time)
    return result


def run(instances, save_file):
    f = open(save_file, 'w')
    pool= Pool(processes=thread_num)
    results=[]
    for k in range(len(instances)):
        data = instances[k]['data']
        score = instances[k]['score']
        result=pool.apply_async(generate_response,(data,score))
        results.append(result)
    pool.close()
    pool.join()
    to_file = []
    for result in results:
        response = result.get()
        if (len(response)>0):
            to_file.append(response)
    json.dump(obj=to_file, fp=f, indent=4)