import networkx as nx

# 读取 GraphML 文件
graph = nx.read_graphml('path/to/your/file.graphml')

# 设置默认值或条件
default_h_score_mean = None  # 或者使用其他表示缺失值的标记

# 为每个节点添加新属性 'H_scoreMean'，根据条件设置值
for node in graph.nodes():
    # 假设有一个条件函数 check_condition(node) 返回布尔值
    if check_condition(node):
        graph.nodes[node]['H_scoreMean'] = calculate_h_score(node)  # 根据需要计算 H_scoreMean
    else:
        graph.nodes[node]['H_scoreMean'] = default_h_score_mean

# 将修改后的图写回 GraphML 文件
nx.write_graphml(graph, 'path/to/your/modified_file.graphml')

# 示例条件函数和计算函数
def check_condition(node):
    # 在这里定义你的条件逻辑
    # 例如，检查节点是否有某个属性或满足某个条件
    return 'some_attribute' in graph.nodes[node] and graph.nodes[node]['some_attribute'] > 0

def calculate_h_score(node):
    # 在这里定义你的 H_scoreMean 计算逻辑
    # 例如，基于节点的某些属性计算 H_scoreMean
    return graph.nodes[node]['some_attribute'] * 0.1  # 示例计算