import os
import re
import openpyxl
import pandas as pd

from path import work_path

os.chdir(work_path)

file_folder = "download/"

def listTOexcel(item_dict, file_path):
    file_path = file_folder + file_path
    df = pd.DataFrame(item_dict)
    # 将 DataFrame 写入 Excel 文件
    df.to_excel(file_path, sheet_name='Sheet1', index=False)
    return

def excelTOlist(file_path):
    file_path = file_folder + file_path
    sheet = pd.read_excel(file_path, sheet_name='Sheet1')
    # orient='list'：每列作为一个键，值是一个列表。
    # orient='records'：每行作为一个字典，值是一个包含这些字典的列表。
    # orient='index'：每行为一个键，值是一个字典。
    # orient='dict'：每列为一个键，值是一个字典（默认）
    data = sheet.to_dict(orient='records')
    return data


