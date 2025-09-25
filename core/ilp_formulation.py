#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILP公式化模块
使用PuLP实现论文中的ILP模型
"""

import pulp
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from .network_model import NetworkModel


class ILPFormulation:
    """ILP公式化类"""

    def __init__(self, network_model: NetworkModel):
        """
        初始化ILP公式化

        Args:
            network_model: 网络模型实例
        """
        self.network = network_model
        self.problem = None
        self.variables = {}

        # 获取网络参数
        self.nodes = network_model.get_nodes()
        self.edges = network_model.get_edges()
        self.demands = network_model.get_demands()
        self.spectrum_slots = network_model.spectrum_slots

        # 索引映射
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.edge_index = {edge: i for i, edge in enumerate(self.edges)}
        self.demand_index = {i: demand for i, demand in enumerate(self.demands)}

    def create_variables(self) -> None:
        """创建所有决策变量"""

        # 路由变量 x_e_k ∈ {0,1}
        self.variables['x'] = {}
        for k_idx, demand in enumerate(self.demands):
            for edge in self.edges:
                var_name = f"x_{edge[0]}_{edge[1]}_{k_idx}"
                self.variables['x'][(edge, k_idx)] = pulp.LpVariable(
                    var_name, cat='Binary'
                )

        # 频谱分配变量 z_s_k ∈ {0,1}
        self.variables['z'] = {}
        for k_idx, demand in enumerate(self.demands):
            for s in range(1, self.spectrum_slots + 1):
                var_name = f"z_{s}_{k_idx}"
                self.variables['z'][(s, k_idx)] = pulp.LpVariable(
                    var_name, cat='Binary'
                )

        # 边-频隙分配变量 t_e_s_k ∈ {0,1}
        self.variables['t'] = {}
        for k_idx, demand in enumerate(self.demands):
            for edge in self.edges:
                for s in range(1, self.spectrum_slots + 1):
                    var_name = f"t_{edge[0]}_{edge[1]}_{s}_{k_idx}"
                    self.variables['t'][(edge, s, k_idx)] = pulp.LpVariable(
                        var_name, cat='Binary'
                    )

        # 边激活变量 a_e ∈ {0,1} (用于O2)
        self.variables['a'] = {}
        for edge in self.edges:
            var_name = f"a_{edge[0]}_{edge[1]}"
            self.variables['a'][edge] = pulp.LpVariable(
                var_name, cat='Binary'
            )

        # 最大频隙位置变量 p_e ∈ Z⁺ (用于O3)
        self.variables['p_e'] = {}
        for edge in self.edges:
            var_name = f"p_e_{edge[0]}_{edge[1]}"
            self.variables['p_e'][edge] = pulp.LpVariable(
                var_name, lowBound=0, cat='Integer'
            )

        # 全局最大频隙位置变量 p ∈ Z⁺ (用于O3)
        self.variables['p'] = pulp.LpVariable(
            "p", lowBound=0, cat='Integer'
        )

    def create_objective(self, objective_type: str = 'O3') -> None:
        """
        创建目标函数，实现多目标优先级 O3 > O1 > O2

        Args:
            objective_type: 目标类型 ('O1', 'O2', 'O3', 'MULTI')
        """
        if objective_type == 'O1':
            # O1: 最小化总跳数
            self.problem.objective = pulp.lpSum(
                self.variables['x'][(edge, k_idx)]
                for k_idx in range(len(self.demands))
                for edge in self.edges
            )

        elif objective_type == 'O2':
            # O2: 最小化使用的边数
            self.problem.objective = pulp.lpSum(
                self.variables['a'][edge] for edge in self.edges
            )

        elif objective_type == 'O3':
            # O3: 最小化最大频隙位置
            self.problem.objective = self.variables['p']

        elif objective_type == 'MULTI':
            # 多目标优先级: O3 > O1 > O2
            # 使用加权方法实现优先级，权重反映优先级顺序
            total_hops = pulp.lpSum(
                self.variables['x'][(edge, k_idx)]
                for k_idx in range(len(self.demands))
                for edge in self.edges
            )
            total_edges = pulp.lpSum(
                self.variables['a'][edge] for edge in self.edges
            )

            # 优先级权重: p (O3) >> total_hops (O1) >> total_edges (O2)
            # 使用大权重确保优先级
            w3 = 1000  # O3权重
            w1 = 10    # O1权重
            w2 = 1     # O2权重

            self.problem.objective = w3 * self.variables['p'] + w1 * total_hops + w2 * total_edges

        else:
            raise ValueError(f"未知目标类型: {objective_type}")

        # 正确设置目标函数
        self.problem.setObjective(self.problem.objective)

    def add_routing_constraints(self) -> None:
        """添加路由约束"""

        for k_idx, demand in enumerate(self.demands):
            source, target = demand['source'], demand['target']

            # 源点约束: ∑_{e∈δ(o_k)} x_{e,k} ≤ 1 (允许拒绝需求)
            source_edges = [edge for edge in self.edges if source in edge]
            self.problem += pulp.lpSum(
                self.variables['x'][(edge, k_idx)] for edge in source_edges
            ) == 1, f"origin_{k_idx}"

            # 终点约束: ∑_{e∈δ(d_k)} x_{e,k} - ∑_{e∈δ(o_k)} x_{e,k} = 0
            target_edges = [edge for edge in self.edges if target in edge]
            self.problem += pulp.lpSum(
                self.variables['x'][(edge, k_idx)] for edge in target_edges
            ) - pulp.lpSum(
                self.variables['x'][(edge, k_idx)] for edge in source_edges
            ) == 0, f"destination_{k_idx}"

            # 度数约束: ∑_{e∈δ(v)} x_{e,k} ≤ 2, ∀v ∈ V \ {o_k, d_k}
            for node in self.nodes:
                if node != source and node != target:
                    node_edges = [edge for edge in self.edges if node in edge]
                    self.problem += pulp.lpSum(
                        self.variables['x'][(edge, k_idx)] for edge in node_edges
                    ) <= 2, f"degree_{node}_{k_idx}"

            # 传输距离约束: \sum_{e \in E} l_e x_{e}^{k} - \bar{l}_k \sum_{e \in \delta(o_k)} x_{e}^{k} \leq 0, \quad \forall k \in K
            total_distance = pulp.lpSum(
                self.network.get_edge_distance(edge[0], edge[1]) * self.variables['x'][(edge, k_idx)]
                for edge in self.edges
            )
            source_flow = pulp.lpSum(
                self.variables['x'][(edge, k_idx)] for edge in source_edges
            )
            reach_constraint = total_distance - demand['reach'] * source_flow <= 0
            self.problem += reach_constraint, f"reach_{k_idx}"

            # 调试信息：打印传输距离约束
            print(f"需求{k_idx+1} ({source}->{target}) 传输距离约束:")
            print(f"  总距离变量: {total_distance}")
            print(f"  源点流: {source_flow}")
            print(f"  传输限制: {demand['reach']}")
            print(f"  约束: {reach_constraint}")

            # 路径连续性约束（针对小规模网络）
            # 对于6节点网络，生成所有可能的子集约束
            other_nodes = [node for node in self.nodes if node != source and node != target]

            # 生成所有包含源点但不包含目标点的子集
            from itertools import combinations
            for r in range(1, len(other_nodes) + 1):
                for subset_nodes in combinations(other_nodes, r):
                    subset = set(subset_nodes)
                    subset.add(source)  # 确保源点在子集中

                    # 计算割集：子集内部边 vs 子集到外部的边
                    cut_internal_edges = []
                    cut_external_edges = []

                    for edge in self.edges:
                        u, v = edge
                        u_in_subset = u in subset
                        v_in_subset = v in subset

                        if u_in_subset and v_in_subset:
                            cut_internal_edges.append(edge)
                        elif u_in_subset or v_in_subset:
                            cut_external_edges.append(edge)

                    # 路径连续性约束: ∑_{e∈δ(X)} x_{e,k} - ∑_{e∈δ(o_k)} x_{e,k} ≥ 0
                    if cut_external_edges:  # 只有当存在割集时才添加约束
                        cut_flow = pulp.lpSum(
                            self.variables['x'][(edge, k_idx)] for edge in cut_external_edges
                        )
                        source_flow_val = pulp.lpSum(
                            self.variables['x'][(edge, k_idx)] for edge in source_edges
                        )
                        path_continuity_constraint = cut_flow - source_flow_val >= 0
                        self.problem += path_continuity_constraint, f"path_continuity_{k_idx}_{'_'.join(sorted(subset))}"

            # 环路消除约束（针对小规模网络的简化版本）
            # 对于每个需求k和每条边e，防止形成包含该边但不包含源点的环路
            for edge in self.edges:
                u, v = edge
                # 创建子集X = {u}（不包含源点source）
                if u != source and v != source:
                    # 计算割集δ(X)其中X = {u}
                    subset_u = {u}
                    cut_edges_u = [e for e in self.edges if (u in e and v not in e) or (v in e and u not in e)]

                    # 环路消除约束: x_{e,k} ≤ ∑_{e'∈δ(o_k)} x_{e',k}
                    if cut_edges_u:
                        self.problem += (
                            self.variables['x'][(edge, k_idx)] <= source_flow_val,
                            f"cycle_elimination_{k_idx}_{edge[0]}_{edge[1]}"
                        )

    def add_edge_activation_constraints(self) -> None:
        """添加边激活约束（用于O2）"""

        for edge in self.edges:
            for k_idx in range(len(self.demands)):
                # a_e - x_{e,k} ≥ 0
                self.problem += (
                    self.variables['a'][edge] - self.variables['x'][(edge, k_idx)] >= 0,
                    f"edge_activation_{edge[0]}_{edge[1]}_{k_idx}"
                )

            # a_e ≤ ∑_{k∈K} x_{e,k}
            self.problem += (
                self.variables['a'][edge] <= pulp.lpSum(
                    self.variables['x'][(edge, k_idx)] for k_idx in range(len(self.demands))
                ),
                f"edge_activation_sum_{edge[0]}_{edge[1]}"
            )

    def add_spectrum_assignment_constraints(self) -> None:
        """添加频谱分配约束"""

        for k_idx, demand in enumerate(self.demands):
            w_k = demand['slots']
            source = demand['source']
            source_edges = [edge for edge in self.edges if source in edge]

            # 通道选择约束: ∑_{s=w_k}^{s̄} z_{s,k} - ∑_{e∈δ(o_k)} x_{e,k} = 0
            self.problem += (
                pulp.lpSum(
                    self.variables['z'][(s, k_idx)] for s in range(w_k, self.spectrum_slots + 1)
                ) - pulp.lpSum(
                    self.variables['x'][(edge, k_idx)] for edge in source_edges
                ) == 0,
                f"channel_selection_{k_idx}"
            )

            # 禁用频隙约束: ∑_{s=1}^{w_k-1} z_{s,k} = 0
            if w_k > 1:
                self.problem += (
                    pulp.lpSum(
                        self.variables['z'][(s, k_idx)] for s in range(1, w_k)
                    ) == 0,
                    f"forbidden_slot_{k_idx}"
                )

            # 边-频隙约束: ∑_{s∈S} t_{e,s,k} - w_k x_{e,k} = 0
            for edge in self.edges:
                self.problem += (
                    pulp.lpSum(
                        self.variables['t'][(edge, s, k_idx)] for s in range(1, self.spectrum_slots + 1)
                    ) - w_k * self.variables['x'][(edge, k_idx)] == 0,
                    f"edge_slot_{edge[0]}_{edge[1]}_{k_idx}"
                )

                # 频谱连续性与邻接性约束
                for s in range(1, self.spectrum_slots + 1):
                    # x_e^k + ∑_{s'=s}^{min(s+w_k-1, s̄)} z_{s'}^k - t_{e,s}^k ≤ 1
                    sum_z = pulp.lpSum(
                        self.variables['z'][(s_prime, k_idx)]
                        for s_prime in range(s, min(s + w_k, self.spectrum_slots + 1))
                    )
                    self.problem += (
                        self.variables['x'][(edge, k_idx)] + sum_z - self.variables['t'][(edge, s, k_idx)] <= 1,
                        f"continuity_contiguity_{edge[0]}_{edge[1]}_{s}_{k_idx}"
                    )

        # 频谱非重叠约束: ∑_{k∈K} t_{e,s,k} ≤ 1
        for edge in self.edges:
            for s in range(1, self.spectrum_slots + 1):
                self.problem += (
                    pulp.lpSum(
                        self.variables['t'][(edge, s, k_idx)] for k_idx in range(len(self.demands))
                    ) <= 1,
                    f"non_overlapping_{edge[0]}_{edge[1]}_{s}"
                )

    def add_max_slot_constraints(self) -> None:
        """添加最大频隙位置约束（用于O3）"""

        for edge in self.edges:
            for k_idx in range(len(self.demands)):
                for s in range(1, self.spectrum_slots + 1):
                    # s · t_{e,s,k} - p_e ≤ 0
                    self.problem += (
                        s * self.variables['t'][(edge, s, k_idx)] - self.variables['p_e'][edge] <= 0,
                        f"max_slot_{edge[0]}_{edge[1]}_{s}_{k_idx}"
                    )

            # p_e - ∑_{k∈K} ∑_{s∈S} s · t_{e,s,k} ≤ 0
            self.problem += (
                self.variables['p_e'][edge] - pulp.lpSum(
                    s * self.variables['t'][(edge, s, k_idx)]
                    for k_idx in range(len(self.demands))
                    for s in range(1, self.spectrum_slots + 1)
                ) <= 0,
                f"max_slot_bound_{edge[0]}_{edge[1]}"
            )

            # p_e ≤ p
            self.problem += (
                self.variables['p_e'][edge] <= self.variables['p'],
                f"global_max_slot_{edge[0]}_{edge[1]}"
            )

        # p ≤ s̄
        self.problem += (
            self.variables['p'] <= self.spectrum_slots,
            "global_max_slot_bound"
        )

    def build_model(self, objective_type: str = 'O3') -> pulp.LpProblem:
        """
        构建完整的ILP模型

        Args:
            objective_type: 目标类型 ('O1', 'O2', 'O3')

        Returns:
            构建好的ILP问题
        """
        # 创建问题
        self.problem = pulp.LpProblem(f"RSA_ILP_{objective_type}", pulp.LpMinimize)

        # 创建变量
        self.create_variables()

        # 创建目标函数
        self.create_objective(objective_type)

        # 添加约束
        self.add_routing_constraints()
        self.add_edge_activation_constraints()
        self.add_spectrum_assignment_constraints()
        self.add_max_slot_constraints()

        return self.problem

    def get_variable_values(self) -> Dict[str, Any]:
        """获取变量值"""
        if not self.problem or not pulp.LpStatus[self.problem.status] == 'Optimal':
            return {}

        values = {}

        # 获取路由变量值
        values['routing'] = {}
        for (edge, k_idx), var in self.variables['x'].items():
            if pulp.value(var) > 0.5:
                if k_idx not in values['routing']:
                    values['routing'][k_idx] = []
                # 处理不同的边表示方式
                if isinstance(edge, tuple) and len(edge) == 2:
                    edge_tuple = (str(edge[0]), str(edge[1]))
                else:
                    edge_tuple = edge
                values['routing'][k_idx].append(edge_tuple)

        # 获取频谱分配变量值
        values['spectrum'] = {}
        for (s, k_idx), var in self.variables['z'].items():
            if pulp.value(var) > 0.5:
                if k_idx not in values['spectrum']:
                    values['spectrum'][k_idx] = []
                values['spectrum'][k_idx].append(s)

        # 获取边-频隙分配变量值
        values['edge_slot'] = {}
        for (edge, s, k_idx), var in self.variables['t'].items():
            if pulp.value(var) > 0.5:
                if k_idx not in values['edge_slot']:
                    values['edge_slot'][k_idx] = {}
                if edge not in values['edge_slot'][k_idx]:
                    values['edge_slot'][k_idx][edge] = []
                values['edge_slot'][k_idx][edge].append(s)

        # 获取边激活变量值
        values['edge_activation'] = {}
        for edge, var in self.variables['a'].items():
            if pulp.value(var) > 0.5:
                values['edge_activation'][edge] = True

        # 获取最大频隙位置值
        values['max_slot'] = {}
        for edge, var in self.variables['p_e'].items():
            values['max_slot'][edge] = int(pulp.value(var))
        values['global_max_slot'] = int(pulp.value(self.variables['p']))

        return values


# 测试函数
def test_ilp_formulation():
    """测试ILP公式化"""
    print("测试ILP公式化...")

    from .network_model import NetworkModel

    # 创建网络模型
    network = NetworkModel()
    network.create_6node_topology()
    network.create_test_demands()

    # 创建ILP公式化
    ilp = ILPFormulation(network)

    # 测试构建模型
    problem = ilp.build_model('O3')
    print(f"✓ ILP模型构建成功")
    print(f"  - 变量数量: {len(problem.variables())}")
    print(f"  - 约束数量: {len(problem.constraints)}")
    print(f"  - 目标函数: {problem.objective}")

    # 测试不同目标
    for obj in ['O1', 'O2', 'O3']:
        problem = ilp.build_model(obj)
        print(f"✓ {obj}目标模型构建成功")


if __name__ == "__main__":
    test_ilp_formulation()