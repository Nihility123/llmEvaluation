import copy
from pathlib import Path
import threading
import time
from typing import Annotated, Union
import typer
from peft import PeftModelForCausalLM
from transformers import (
    AutoModel,
    AutoTokenizer,
)
import torch
from transformers.generation import StoppingCriteria

import re
import random

from kg import get_kg
from network_search import search

import json
from flask import jsonify

from concurrent.futures import ThreadPoolExecutor

from path import model_dir

# ============================================================ MODEL ============================================================
def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
):
    model_dir = Path(model_dir).expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        import json
        with open(model_dir / 'adapter_config.json', 'r', encoding='utf-8') as file:
            config = json.load(file)
        model = AutoModel.from_pretrained(
            config.get('base_model_name_or_path'),
            trust_remote_code=trust_remote_code,
            device_map='auto',
            torch_dtype=torch.float16
        )
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=model_dir,
            trust_remote_code=trust_remote_code,
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            device_map='auto',
            torch_dtype=torch.float16
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=trust_remote_code,
        encode_special_tokens=True,
        use_fast=False
    )
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_dir)

generate_kwargs = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.9,
    "repetition_penalty": 1.2,
    "eos_token_id": model.config.eos_token_id,
}

def que(system, prompt):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        use_cache=False,  # 禁用缓存状态
    ).to(model.device)
    # 使用自定义的停止条件
    outputs = model.generate(
        **inputs,
        stopping_criteria=[stop_criteria],  # 将自定义停止条件传入
        **generate_kwargs
    )
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()
    return response

# ============================================================ CONTROL ============================================================

STOP_SIGN = True
need_check = True

class StopGenerationCriteria(StoppingCriteria):
    def __init__(self):
        self.stop_generation = False

    def set_stop(self):
        self.stop_generation = True

    def set_allow(self):
        self.stop_generation = False

    def is_stoped(self):
        return self.stop_generation

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_generation
    
stop_criteria = StopGenerationCriteria()

def allow_gen():
    global STOP_SIGN
    STOP_SIGN = False
    stop_criteria.set_allow()

def stop_gen():
    global STOP_SIGN
    STOP_SIGN = True
    stop_criteria.set_stop()

def is_generating():
    return not STOP_SIGN or not stop_criteria.is_stoped()

stop_gen()

# ============================================================ DATA ============================================================
num_of_sentences = 0
aaa = "xxx"
bbb = "xxx"
ccc = True
ddd = True
number = 20
search_text_list = []
usable_num = 0
generating_text = "生成进行中"
prompts_masked = [
    "围绕“***”，5个甲类句子",
]
prompts_masked_text = [
    "围绕文本，5个甲类句子",
]
prompts_masked_kg = "根据文本，5个甲类句子"
bbb_masked = "，和###相关"
prefix_search = "根据文本，"
que_num = "10"
prefix_anti = "反对XXX，"
answer_text = ""

# ============================================================ FUNCTION ============================================================
def prompt_handle():
    prompts_list = []
    if ccc == True:
        for pro in prompts_masked:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的甲类")
            else:
                pro = pro.replace("甲类", "简短的甲类")
            prompts_list.append(prefix_anti + pro.replace("5", f"{que_num}").replace("***", aaa) + bbb_masked.replace("###", bbb))
    else:
        for pro in prompts_masked:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的乙类")
            else:
                pro = pro.replace("甲类", "简短的乙类")
            prompts_list.append(prefix_anti.replace("反对", "") + pro.replace("5", f"{que_num}").replace("***", aaa) + bbb_masked.replace("###", bbb))
    return prompts_list

def prompt_handle_text(text):
    prompts_list = []
    if ccc == True:
        for pro in prompts_masked_text:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的甲类")
            else:
                pro = pro.replace("甲类", "简短的甲类")
            prompts_list.append(prefix_anti + pro.replace("5", f"{que_num}") + bbb_masked.replace("###", bbb) + "：\n" + text)
    else:
        for pro in prompts_masked_text:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的乙类")
            else:
                pro = pro.replace("甲类", "简短的乙类")
            prompts_list.append(prefix_anti.replace("反对", "") + pro.replace("5", f"{que_num}") + bbb_masked.replace("###", bbb) + "：\n" + text)
    return prompts_list

def prompt_handle_text_new(text):
    prompts_list = []
    if ccc == True:
        for pro in prompts_masked:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的甲类")
            else:
                pro = pro.replace("甲类", "简短的甲类")
            prompts_list.append(prefix_anti + prefix_search + pro.replace("5", f"{que_num}").replace("***", aaa) + bbb_masked.replace("###", bbb) + "：\n" + text)
    else:
        for pro in prompts_masked:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的乙类")
            else:
                pro = pro.replace("甲类", "简短的乙类")
            prompts_list.append(prefix_anti.replace("反对", "") + prefix_search + pro.replace("5", f"{que_num}").replace("***", aaa) + bbb_masked.replace("###", bbb) + "：\n" + text)
    return prompts_list

def prompt_handle_kg_new(content_list):
    text = ""
    text_list = random.sample(content_list, min(30, len(content_list)))
    for tt in text_list:
        text += tt
    
    prompts_list = []
    if ccc == True:
        for pro in prompts_masked:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的甲类")
            else:
                pro = pro.replace("甲类", "简短的甲类")
            prompts_list.append(prefix_anti + prefix_search + pro.replace("5", f"{que_num}").replace("***", aaa) + bbb_masked.replace("###", bbb) + "：\n" + text)
    else:
        for pro in prompts_masked:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的乙类")
            else:
                pro = pro.replace("甲类", "简短的乙类")
            prompts_list.append(prefix_anti.replace("反对", "") + prefix_search + pro.replace("5", f"{que_num}").replace("***", aaa) + bbb_masked.replace("###", bbb) + "：\n" + text)
    return prompts_list

def prompt_handle_kg(text_list):
    prompts_list = []
    if ccc == True:
        if ddd == True:
            pro = prompts_masked_kg.replace("甲类", "复杂的甲类")
        else:
            pro = prompts_masked_kg.replace("甲类", "简短的甲类")
        prefix = prefix_anti + pro.replace("5", f"{que_num}") + bbb_masked.replace("###", bbb) + "：\n"
        for text in text_list:
            prompts_list.append(prefix + text)
    else:
        if ddd == True:
            pro = prompts_masked_kg.replace("甲类", "复杂的乙类")
        else:
            pro = prompts_masked_kg.replace("甲类", "简短的乙类")
        prefix = prefix_anti.replace("反对", "") + pro.replace("5", f"{que_num}") + bbb_masked.replace("###", bbb) + "：\n"
        for text in text_list:
            prompts_list.append(prefix + text)
    return prompts_list

def prompt_handle_aaa_text(text_list):
    prompts_list = []
    if ccc == True:
        for pro in prompts_masked:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的甲类")
            else:
                pro = pro.replace("甲类", "简短的甲类")
            prompts_list.append(prefix_anti + prefix_search + pro.replace("5", f"{que_num}").replace("***", aaa) + bbb_masked.replace("###", bbb) + "：\n" + random.choice(text_list))
    else:
        for pro in prompts_masked:
            if ddd == True:
                pro = pro.replace("甲类", "复杂的乙类")
            else:
                pro = pro.replace("甲类", "简短的乙类")
            prompts_list.append(prefix_anti.replace("反对", "") + prefix_search + pro.replace("5", f"{que_num}").replace("***", aaa) + bbb_masked.replace("###", bbb) + "：\n" + random.choice(text_list))
    return prompts_list

def prompt_handle_aaa_text_2(text_list):
    prompts_list = []
    if ccc == True:
        for pro in prompts_masked:
            prompts_list.append("反对XXX，" + prefix_search + pro.replace("甲类", "复杂的甲类").replace("5", f"{que_num}").replace("***", aaa) + "：\n" + random.choice(text_list))
    else:
        for pro in prompts_masked:
            prompts_list.append("XXX，" + prefix_search + pro.replace("5", f"{que_num}").replace("***", aaa) + "：\n" + random.choice(text_list))
    return prompts_list

def prompt_handle_multi_turn_first(iaaa):
    prompts_list = []
    for pro in prompts_masked:
        pro = pro.replace("甲类", "简短的乙类")
        prompts_list.append(pro.replace("5", f"{que_num}").replace("***", iaaa))
    return prompts_list

def prompt_handle_multi_turn_first_text(itext):
    prompts_list = []
    for pro in prompts_masked_text:
        pro = pro.replace("甲类", "简短的乙类")
        prompts_list.append(pro.replace("5", f"{que_num}") + "：\n" + itext)
    return prompts_list

# �
def contains_lowercase_or_phrase(s, phrase="�"):
    # 检查字符串中是否包含小写字母或指定的词组
    lowercase_pattern = r'[a-z]'
    # phrase_pattern = re.escape(phrase)  # 转义词组中的特殊字符
    return bool(re.search(lowercase_pattern, s)) or phrase in s

def answer_handle(prompts_list, check_english, last_num_of_sentences):
    global aaa
    global num_of_sentences
    # 存储合格句子
    # answers_list = []
    answers_list = []
    # 遍历 prompts_masked
    i = 0
    # 生成句子计数
    cou = 0
    num_of_sentences = last_num_of_sentences
    # 是否达到目标数量
    is_suitable = False
    while (is_suitable == False):
        ques = prompts_list[i]
        # print(ques)
        i = (i + 1) % len(prompts_list)
        
        # 手动中止
        if STOP_SIGN:
            return []
        response = que("", ques)
        # 手动中止
        if STOP_SIGN:
            return []
        
        response = re.sub("^\n+|\n+$", "", response)
        response = re.sub(r"\n\s*\n", "\n", response)
        # print(response)

        for line in response.splitlines():
            if line == "":
                continue
            # 不-含小写英文 且 生成句子有小写英文，舍弃
            if check_english == True:
                if (not contains_lowercase_or_phrase(aaa)) and contains_lowercase_or_phrase(line):
                    # print("        <舍弃> "+line)
                    continue
            # print(line)
            line = line.replace("\"", "").strip(" ")
            if (line[0] <= '9' and line[0] >= '0'):
                if (line[1] != "."):
                    # print("        <舍弃> "+line)
                    continue
                print(f"保留-->>    {line}")
                answers_list.append(line[2:].strip(" "))
                cou += 1
                num_of_sentences += 1
                # 达到数量，直接退出
                if cou >= number:
                    is_suitable = True
                    break
            # else:
            #     print("        <舍弃> "+line)
        # if is_suitable:
        #     break
    return answers_list

# ============================================================ GENERATION ============================================================
def gen_que(iaaa, ibbb, iccc, iddd, inumber, kg, begin_num_of_sentences):
    global need_check
    if need_check and is_generating():
        # ggg = is_generating()
        # print(f"is_generating: {ggg}")
        return {"title": generating_text, "sentences": []}
    allow_gen()
    global bbb, aaa, ccc, ddd, number
    global answer_text

    aaa = iaaa
    bbb = ibbb
    ccc = iccc
    ddd = iddd
    number = inumber

    if kg == True:
        results = search_kg(0)
        if results["title"] != "":
            return results
        else: 
            kg = False

    prompts = prompt_handle()    
    answers = answer_handle(prompts, True, begin_num_of_sentences)
    random.shuffle(answers)

    answer_text = ""
    answer_text = f"“{aaa}”“{bbb}”{number}："
    # 取前 number 个
    return {"title": answer_text, "sentences": answers[:number]}

def search_kg(begin_num_of_sentences):
    global answer_text

    # content_list = []
    content_list = get_kg(aaa)

    # 知识图谱中不存在相关内容
    # if len(content_list) == 0:
    # if content_text == "":
    if len(content_list) == 0:
        print(f"知识图谱无 {aaa}")
        return {"title": "", "sentences": []}
    
    # 有多少content就写多少prompt
    # prompts = prompt_handle_kg(content_list)
    prompts = prompt_handle_kg_new(content_list)
    answers = answer_handle(prompts, True, begin_num_of_sentences)
    random.shuffle(answers)
    
    answer_text = ""
    answer_text = f"“{aaa}”“{bbb}”{number}"
    # for i in range(0, min(number, len(answers))):
    #     answer_text += f"{i + 1}. {answers[i]}\n"
    # print(answer_text)
    # return answer_text
    return {"title": answer_text, "sentences": answers[:number]}

def gen_via_text(text, ibbb, iccc, iddd, inumber, begin_num_of_sentences):
    global need_check
    if need_check and is_generating():
        return {"title": generating_text, "sentences": []}
    allow_gen()
    global ccc, bbb, ddd, number, aaa
    global answer_text

    ccc = iccc
    bbb = ibbb
    ddd = iddd
    number = inumber
    aaa = text
    
    prompts = prompt_handle_text(text)
    answers = answer_handle(prompts, True, begin_num_of_sentences)
    random.shuffle(answers)

    answer_text = ""
    answer_text = f"“{bbb}”{number}"
    # for i in range(0, min(number, len(answers))):
    #     answer_text += f"{i + 1}. {answers[i]}\n"
    # print(answer_text)
    # return answer_text
    return {"title": answer_text, "sentences": answers[:number]}

def gen_via_text_new(iaaa, text, ibbb, iccc, iddd, inumber, begin_num_of_sentences):
    global need_check
    if need_check and is_generating():
        return {"title": generating_text, "sentences": []}
    allow_gen()
    global ccc, bbb, ddd, number, aaa
    global answer_text

    ccc = iccc
    bbb = ibbb
    ddd = iddd
    number = inumber
    aaa = iaaa
    
    prompts = prompt_handle_text_new(text)
    answers = answer_handle(prompts, True, begin_num_of_sentences)
    random.shuffle(answers)

    answer_text = ""
    answer_text = f"“{bbb}”{number}"
    # for i in range(0, min(number, len(answers))):
    #     answer_text += f"{i + 1}. {answers[i]}\n"
    # print(answer_text)
    # return answer_text
    return {"title": answer_text, "sentences": answers[:number]}

def gen_use_search(iaaa, ibbb, iccc, term, iddd, inumber, begin_num_of_sentences, research=True):
    global need_check
    if need_check and is_generating():
        return {"title": generating_text, "sentences": []}
    allow_gen()
    global bbb, aaa, ccc, ddd, number, search_text_list, usable_num
    global answer_text

    aaa = iaaa
    bbb = ibbb
    ccc = iccc
    ddd = iddd
    number = inumber

    if term == "":
        term = aaa
    
    # 仅需要重新搜索时再重新搜索，否则复用
    if research:
        search_text_list = search(term)
        usable_num = 0
        for text in search_text_list:
            if text != "":
                usable_num += 1

    # if len(text_list) > 0:
    #     prompts = prompt_handle_aaa_text(text_list)
    # else:
    #     prompts = prompt_handle()

    # 搜索失败
    if len(search_text_list) == 0:
        return {"title": "搜索失败，请重试", "sentences": []}

    prompts = prompt_handle_aaa_text(search_text_list)
    answers = answer_handle(prompts, True, begin_num_of_sentences)
    random.shuffle(answers)

    answer_text = ""
    answer_text = f"搜索成功，找到 {len(search_text_list)} 条，成功访问 {usable_num} 条\n‘{term}’“{aaa}”“{bbb}”{number}"
    # for i in range(0, min(number, len(answers))):
    #     answer_text += f"{i + 1}. {answers[i]}\n"
    # print(answer_text)
    # return answer_text
    return {"title": answer_text, "sentences": answers[:number]}

def gen_multi_turn(type, iaaa, itext, ibbb, iccc, term, iddd, inumber, kg):
    global need_check
    if need_check and is_generating():
        return {"title": generating_text, "sentences": []}
    allow_gen()
    global number, num_of_sentences
    number = inumber

    # 第一轮
    print(f"<<<<<<<<<< 第一轮 {inumber} >>>>>>>>>>")
    prompts_1 = []
    match type:
        case "text":
            prompts_1 = prompt_handle_multi_turn_first_text(itext)
        case _:
            prompts_1 = prompt_handle_multi_turn_first(iaaa)
    answers_1 = answer_handle(prompts_1, True, 0) # begin_num_of_sentences = 0
    answers_1 = answers_1[:inumber]

    answers_final = []
    
    # 出错，第一轮没有输出句子，返回空值
    if len(answers_1) == 0:
        answers_final.append("失败，请重试")
    #     return {"title": "失败，请重试", "sentences": []}

    for i in range(len(answers_1)):
        ans_1 = answers_1[i]
        # 第二轮
        print(f"<<<<<<<<<< 第二轮 {i+1}/{inumber} >>>>>>>>>>")
        # 第一轮的回答
        # 手动中止

        if STOP_SIGN:
            return {"title": "", "sentences": []}
        response_1 = que("", ans_1 + "给出简短的回答").replace("\n", "")
        # 手动中止
        if STOP_SIGN:
            return {"title": "", "sentences": []}
        
        new_aaa = ans_1 + response_1
        data = {"title": "", "sentences": []}

        need_check = False
        match type:
            case "text":
                data = gen_via_text_new(new_aaa, itext, ibbb, False, iddd, 1, num_of_sentences)
            case "search":
                data = gen_use_search(new_aaa, ibbb, False, term, iddd, 1, num_of_sentences, True if i == 0 else False) # 仅第一次需要搜索，后面复用
            case "aaa" | _: 
                data = gen_que(new_aaa, ibbb, False, iddd, 1, kg, num_of_sentences) # iccc, ddd, number ：1个复杂乙类句子
        need_check = True

        answers_2 = data["sentences"]
        # 出错，第二轮没有输出句子，只返回第一轮句子
        if len(answers_2) == 0:
            wrong_text = "出错，只输出一轮。"
            answers_final.append(f"{wrong_text}\n(1){ans_1}\n    【回答】：{response_1}")
        else: # 第二轮没句子
            ans_2 = answers_2[0]
            # 乙类只有两轮，直接结束
            # if iccc == False:
            #     answers_final.append(f"(1){ans_1}\n    【回答】：{response_1}\n(2){ans_2}")
            # else: # 甲类
            # 第三轮
            print(f"<<<<<<<<<< 第三轮 {i+1}/{inumber} >>>>>>>>>>")

            # 手动中止
            if STOP_SIGN:
                return {"title": "", "sentences": []}
            response_2 = que("", ans_2 + "给出简短的回答").replace("\n", "")
            # 手动中止
            if STOP_SIGN:
                return {"title": "", "sentences": []}
            
            new_text = new_aaa + ans_2 + response_2

            need_check = False
            data_2 = gen_via_text(new_text, ibbb, iccc, iddd, 1, num_of_sentences) # iccc, ddd, number ：1个复杂甲类句子
            need_check = True

            answers_3 = data_2["sentences"]
            # 出错，第三轮没有输出句子，只返回第一、二轮句子
            if len(answers_2) == 0:
                # wrong_text = "出错，只输出一、二轮。"
                answers_final.append(f"出错，只输出一、二轮。\n{wrong_text}\n(1){ans_1}\n    【回答】：{response_1}\n(2){ans_2}\n    【回答】：{response_2}")
            else: # 第三轮没句子
                ans_3 = answers_3[0]
                answers_final.append(f"(1){ans_1}\n    【回答】：{response_1}\n(2){ans_2}\n    【回答】：{response_2}\n(3){ans_3}")
    
    answer_text = ""
    match type:
        case "1":
            answer_text = f"“{ibbb}”的{inumber}"
        case "2":
            answer_text = f"‘{term}’“{iaaa}”“{ibbb}”{inumber}"
        case "3" | _:
            answer_text = f"“{iaaa}”“{ibbb}”{inumber}"
    
    return {"title": answer_text, "sentences": answers_final}

def gen_use_search_2(iaaa, ibbb, iccc, iddd, inumber, begin_num_of_sentences, research=False):
    if is_generating():
        return {"title": generating_text, "sentences": []}
    allow_gen()
    global bbb, aaa, ccc, ddd, number, search_text_list, usable_num
    global answer_text

    aaa = iaaa
    bbb = ibbb
    ccc = iccc
    ddd = iddd
    number = inumber

    # if term == "":
    #     term = aaa
    
    # 仅需要重新搜索时再重新搜索，否则复用
    # if research:
    #     search_text_list = search(term)
    #     usable_num = 0
    #     for text in search_text_list:
    #         if text != "":
    #             usable_num += 1

    # if len(text_list) > 0:
    #     prompts = prompt_handle_aaa_text(text_list)
    # else:
    #     prompts = prompt_handle()

    # 搜索失败
    if len(search_text_list) == 0:
        return {"title": "搜索失败，请重试", "sentences": []}

    prompts = prompt_handle_aaa_text(search_text_list)
    answers = answer_handle(prompts, True, begin_num_of_sentences)
    random.shuffle(answers)

    answer_text = ""
    answer_text = f"搜索成功，找到 {len(search_text_list)} 条，成功访问 {usable_num} 条\n‘{term}’“{aaa}”“{bbb}”{number}："
    # for i in range(0, min(number, len(answers))):
    #     answer_text += f"{i + 1}. {answers[i]}\n"
    # print(answer_text)
    # return answer_text
    return {"title": answer_text, "sentences": answers[:number]}

def gen_use_search_2(iaaa, iccc, inumber):
    global bbb, aaa, ccc, number
    global answer_text

    aaa = iaaa
    ccc = iccc
    number = inumber

    text_list = search(iaaa)
    if len(text_list) > 0:
        prompts = prompt_handle_aaa_text_2(text_list)
    else:
        prompts = prompt_handle()
    answers = answer_handle(prompts)
    random.shuffle(answers)

    answer_text = ""
    answer_text = f"搜索，“{aaa}”{number}：\n"
    for i in range(0, min(number, len(answers))):
        answer_text += f"{i + 1}. {answers[i]}\n"
    print(answer_text)
    return answer_text





