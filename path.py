from pathlib import Path

# 获取当前脚本所在的目录
script_dir = Path(__file__).parent.absolute()

# 工作目录
work_path = str(script_dir) + "/"
print(work_path)

# 生成模型
model_dir = str(script_dir.parent.absolute() / "my_model_path")
# 分类模型
model_path_field = str(script_dir.parent.absolute() / "my_classification_path_1")
model_path_induced = str(script_dir.parent.absolute() / "my_classification_path_2")

# Neo4j连接信息
neo4j_uri = "http://localhost:7474"  # Neo4j URI
neo4j_username = "neo4j"  # Neo4j用户名
neo4j_password = "neo4j"  # Neo4j密码













