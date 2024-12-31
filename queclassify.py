import torch
from transformers import BertTokenizer, BertForSequenceClassification
from mytools import listTOexcel, excelTOlist

from path import model_path_A, model_path_a

label_id_dict_A = {"A": 0, "B": 1, "C": 2, "D": 3}
label_id_dict_a = {"a": 0, "b": 1}
id_label_dict_A = {}
id_label_dict_a = {}

for item in label_id_dict_A:
    id_label_dict_A[label_id_dict_A[item]] = item
for item in label_id_dict_a:
    id_label_dict_a[label_id_dict_a[item]] = item

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")

# 加载模型和分词器
tokenizer_A = BertTokenizer.from_pretrained(model_path_A)
model_A = BertForSequenceClassification.from_pretrained(model_path_A)
model_A.to(device)
# 设置模型为评估模式
model_A.eval()

# 加载模型和分词器
tokenizer_a = BertTokenizer.from_pretrained(model_path_a)
model_a = BertForSequenceClassification.from_pretrained(model_path_a)
model_a.to(device)
# 设置模型为评估模式
model_a.eval()

top_n = 5

file_path = "items.xlsx"

def classify(text, type):
    match type:
        case "ABCD":
            model = model_A
            tokenizer = tokenizer_A
            id_label_dict = id_label_dict_A
            i_top = top_n
        case "abcd" | _:
            model = model_a
            tokenizer = tokenizer_a
            id_label_dict = id_label_dict_a
            i_top = 1

    # 对输入文本进行编码
    encoded_dict = tokenizer.encode_plus(
        text,
        # 是否添加特殊标记，[sep]和[cls]是bert两种特殊标记
        add_special_tokens=True,
        # 文本截断
        max_length=100,
        # 启用截断
        truncation=True,
        # 启用填充
        padding='max_length',
        # 返回注意力掩码
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    # 在需要的话将输入移动到GPU上
    # device = torch.device("cuda:0")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 使用模型进行预测
    # 关闭自动求导
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # 获取预测结果的概率分布
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    top_probabilities, top_indices = torch.topk(probabilities, i_top, dim=1)

    list0 = top_indices.tolist()[0]
    list1 = []
    list2 = top_probabilities.tolist()[0]
    for item in list0:
        list1.append(id_label_dict[item])

    result = []
    for i in range(i_top):
        result.append([list1[i], list2[i]])

    print(result)
    return result

# 单个元素
def classify_item(text):
    result_A = classify(text, "ABCD")
    result_a = classify(text, "abcd")
    # print(result_list)
    return {"ABCD": result_A, "abcd": result_a}

# 多个元素
def classify_items(text_list):
    result_list = []
    for text in text_list:
        result_list.append(classify_item(text))
    # print(result_list)
    return result_list

def classify_file():
    # 文件转成列表 ——> 分类 ——> 写回文件
    item_list = []
    data = excelTOlist(file_path)
    for item in data:
        item_list.append(item["xxx"])
    results = classify_items(item_list)
    print(results)

    for i in range(len(data)):
        print(f"============================={results[i]}")
        data[i]["预测a"] = results[i]["abcd"][0][0]
        data[i]["预测a置信度"] = results[i]["abcd"][0][1]
        for j in range(top_n):
            data[i][f"预测A{j+1}"] = results[i]["ABCD"][j][0]
            data[i][f"预测A{j+1}置信度"] = results[i]["ABCD"][j][1]
    listTOexcel(data, file_path)
    return

def calculate_acc():
    data = excelTOlist(file_path)
    all_num = len(data)
    right_A_num = [0, 0, 0]
    right_a_num = 0
    result = {
        "ABCD_acc": [-0.01, -0.01, -0.01],
        "abcd_acc": -0.01,
    }
    for item in data:
        try:
            real_A = item["A"]
            real_a = item["a"]
            predict_A_list = []
            for i in range(top_n):
                predict_A_list.append(item[f"预测A{i+1}"])
            predict_a = item["预测a"]
        except KeyError: # 错误，有 'key' 不存在，无法计算（不准确）
            return result

        # ABCD
        # top1
        if real_A in predict_A_list[0]:
            right_A_num[0] += 1
            right_A_num[1] += 1
            right_A_num[2] += 1
        # top3
        elif real_A in predict_A_list[:3]:
            right_A_num[1] += 1
            right_A_num[2] += 1
        # top5
        elif real_A in predict_A_list[:3]:
            right_A_num[2] += 1

        # abcd
        if real_a == predict_a:
            right_a_num += 1

    for i in range(len(right_A_num)):
        result["ABCD_acc"][i] = right_A_num[i] / all_num
    result["abcd_acc"] = right_a_num / all_num
    return result

    


