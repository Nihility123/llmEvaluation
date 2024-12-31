
import importlib
import os
import pandas as pd
from py2neo import Graph, Node, Relationship
# 将JSON文本解析为Python对象
import json

from path import work_path, neo4j_uri, neo4j_username, neo4j_password

# 连接到Neo4j数据库
graph = Graph(neo4j_uri, auth=(neo4j_username, neo4j_password))
# # 清空所有内容
# graph.delete_all()

# 导入neo4j知识图谱代码
# 文件夹路径
os.chdir(work_path)
folder_path = "kg_path"
# 存储加载的模块
modules = {}
# 动态加载所有 .py 文件
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            # 提取模块名称（去掉扩展名）
            module_name = filename[:-3]
            # print(module_name)
            
            if module_name == "__init__":
                continue

            # 动态加载模块
            full_module_name = f"{folder_path}.{module_name}"
            module = importlib.import_module(full_module_name)
            
            # 保存模块到字典中
            modules[module_name] = module
            
            print(f"Loaded module: {full_module_name}")
else:
    print(f"Folder '{folder_path}' does not exist or is not a directory.")


# 加载 知识图谱 到 neo4j
def load(kg_name):
    if kg_name in modules:  # 如果有一个 kg_name.py
        module = modules[kg_name]
        # 假设模块内有 data
        if hasattr(module, "data"):  
            data = getattr(module, "data")

            # 清空所有内容
            graph.delete_all()
            graph_data = json.loads(data)

            # 创建节点
            for node_data in graph_data['nodes']:
                node = Node(node_data['label'], name=node_data['id'])
                graph.create(node)

            # 创建关系
            for edge_data in graph_data['edges']:
                source = edge_data['source']
                target = edge_data['target']
                relation_type = edge_data['label']

                query = (
                    f"MATCH (a), (b) WHERE a.name = '{source}' AND b.name = '{target}' "
                    f"CREATE (a)-[:{relation_type}]->(b)"
                )
                graph.run(query)

            print("Graph creation completed.")
            return "1"
    return "0"


# 获取 知识图谱 文本
def get_kg(kg_name):
    if kg_name in modules:  # 如果有一个 kg_name.py
        module = modules[kg_name]
        # 假设模块内有 data
        if hasattr(module, "data"):  
            data = getattr(module, "data")
            graph_data = json.loads(data)
            # content_list = ""
            content_list = []

            # 查找 + 拼接 关系
            for edge_data in graph_data['edges']:
                source = edge_data['source']
                target = edge_data['target']
                relation_type = edge_data['label']
                # content_list += f"{source}{relation_type}{target}，"
                content_list.append(f"{source}{relation_type}{target}，")
            return content_list
    return []

def get_kg_list():
    return list(modules.keys())

