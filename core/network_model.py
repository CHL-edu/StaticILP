#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络模型核心模块
管理网络拓扑和需求数据
"""

import networkx as nx
import json
import os
from typing import Dict, List, Tuple, Any, Optional


class NetworkModel:
    """网络模型类，管理拓扑和需求数据"""

    def __init__(self):
        """初始化网络模型"""
        self.graph = nx.Graph()
        self.demands = []
        self.spectrum_slots = 10
        self.node_list = []
        self.edge_list = []

    def create_6node_topology(self) -> nx.Graph:
        """
        创建6点8边的测试网络拓扑
        节点: a, b, c, d, e, f
        边和距离: (a,b,1), (b,c,1), (c,f,2), (a,f,2), (c,d,1), (d,e,1), (f,e,1), (f,d,3)
        """
        self.graph = nx.Graph()

        # 添加节点
        self.node_list = ['a', 'b', 'c', 'd', 'e', 'f']
        self.graph.add_nodes_from(self.node_list)

        # 添加边及其属性
        edges_with_attrs = [
            ('a', 'b', {'distance': 1}),
            ('b', 'c', {'distance': 1}),
            ('c', 'f', {'distance': 2}),
            ('a', 'f', {'distance': 2}),
            ('c', 'd', {'distance': 1}),
            ('d', 'e', {'distance': 1}),
            ('f', 'e', {'distance': 1}),
            ('f', 'd', {'distance': 3}),
        ]

        self.graph.add_edges_from(edges_with_attrs)
        self.edge_list = list(self.graph.edges())

        # 为每条边初始化频谱资源
        self.spectrum_slots = 10

        for u, v, data in self.graph.edges(data=True):
            data['total_spectrum'] = self.spectrum_slots
            data['used'] = 0.0
            data['available_spectrum'] = list(range(self.spectrum_slots))
            data['used_spectrum'] = set()

        return self.graph

    def create_test_demands(self) -> List[Dict[str, Any]]:
        """
        创建测试需求
        返回: 需求列表，每个需求包含source, target, slots, reach
        """
        demands = [
            {'source': 'a', 'target': 'c', 'slots': 2, 'reach': 4},
            {'source': 'a', 'target': 'd', 'slots': 1, 'reach': 4},
            {'source': 'b', 'target': 'f', 'slots': 2, 'reach': 4},
            {'source': 'b', 'target': 'e', 'slots': 1, 'reach': 4},
            {'source': 'd', 'target': 'f', 'slots': 3, 'reach': 4}
        ]

        self.demands = demands
        return demands

    def get_node_index(self, node: str) -> int:
        """获取节点索引"""
        return self.node_list.index(node)

    def get_edge_index(self, u: str, v: str) -> int:
        """获取边索引"""
        return self.edge_list.index((u, v)) if (u, v) in self.edge_list else self.edge_list.index((v, u))

    def get_nodes(self) -> List[str]:
        """获取节点列表"""
        return self.node_list

    def get_edges(self) -> List[Tuple[str, str]]:
        """获取边列表"""
        return self.edge_list

    def get_demands(self) -> List[Dict[str, Any]]:
        """获取需求列表"""
        return self.demands

    def get_edge_distance(self, u: str, v: str) -> float:
        """获取边距离"""
        return self.graph.edges[u, v]['distance']

    def validate_network_connectivity(self) -> bool:
        """验证网络连通性"""
        return nx.is_connected(self.graph)

    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_demands': len(self.demands),
            'total_spectrum_slots': self.spectrum_slots,
            'avg_node_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'network_diameter': nx.diameter(self.graph),
            'avg_shortest_path_length': nx.average_shortest_path_length(self.graph, weight='distance'),
            'is_connected': nx.is_connected(self.graph)
        }

    def save_to_json(self, filename: str = "E:/PythonProject/ILP/log/network_model.json") -> None:
        """保存网络模型到JSON文件"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        data = {
            'nodes': self.node_list,
            'edges': [],
            'demands': self.demands,
            'spectrum_slots': self.spectrum_slots
        }

        for u, v, attr in self.graph.edges(data=True):
            data['edges'].append({
                'source': u,
                'target': v,
                'distance': attr['distance'],
                'total_spectrum': attr['total_spectrum']
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_json(self, filename: str) -> None:
        """从JSON文件加载网络模型"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.node_list = data['nodes']
        self.demands = data['demands']
        self.spectrum_slots = data['spectrum_slots']

        # 重建图
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.node_list)

        for edge_data in data['edges']:
            u, v = edge_data['source'], edge_data['target']
            distance = edge_data['distance']
            total_spectrum = edge_data['total_spectrum']

            self.graph.add_edge(u, v, distance=distance, total_spectrum=total_spectrum,
                              used=0.0, available_spectrum=list(range(total_spectrum)),
                              used_spectrum=set())

        self.edge_list = list(self.graph.edges())

    def print_network_info(self) -> None:
        """打印网络信息"""
        stats = self.get_network_stats()
        print("=== 网络拓扑信息 ===")
        print(f"节点数: {stats['num_nodes']}")
        print(f"边数: {stats['num_edges']}")
        print(f"需求个数: {stats['num_demands']}")
        print(f"频隙数: {stats['total_spectrum_slots']}")
        print(f"平均节点度: {stats['avg_node_degree']:.2f}")
        print(f"网络直径: {stats['network_diameter']}")
        print(f"平均最短路径长度: {stats['avg_shortest_path_length']:.2f}")
        print(f"连通性: {'连通' if stats['is_connected'] else '不连通'}")

        print("\n边信息:")
        for u, v, data in self.graph.edges(data=True):
            print(f"  ({u}, {v}): 距离={data['distance']}, 频隙数={data['total_spectrum']}")

        print("\n需求信息:")
        for i, demand in enumerate(self.demands):
            print(f"  需求{i+1}: {demand['source']}->{demand['target']}, "
                  f"频隙={demand['slots']}, 传输距离限制={demand['reach']}")


# 测试函数
def test_network_model():
    """测试网络模型功能"""
    print("测试网络模型...")

    # 创建网络模型
    model = NetworkModel()
    model.create_6node_topology()
    model.create_test_demands()

    # 打印信息
    model.print_network_info()

    # 验证连通性
    assert model.validate_network_connectivity(), "网络应该连通"

    # 保存和加载测试
    model.save_to_json()

    new_model = NetworkModel()
    new_model.load_from_json("E:/PythonProject/ILP/log/network_model.json")

    assert len(new_model.get_nodes()) == len(model.get_nodes()), "节点数应该相同"
    assert len(new_model.get_edges()) == len(model.get_edges()), "边数应该相同"
    assert len(new_model.get_demands()) == len(model.get_demands()), "需求数应该相同"

    print("✓ 网络模型测试通过")


if __name__ == "__main__":
    test_network_model()