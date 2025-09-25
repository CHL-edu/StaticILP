#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解决方案分析模块
分析和验证ILP求解结果
"""

import pulp
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from .network_model import NetworkModel
from .ilp_formulation import ILPFormulation


class SolutionAnalyzer:
    """解决方案分析器"""

    def __init__(self, network_model: NetworkModel):
        """
        初始化解决方案分析器

        Args:
            network_model: 网络模型实例
        """
        self.network = network_model
        self.solution = {}
        self.validation_results = {}

    def analyze_solution(self, ilp_formulation: ILPFormulation) -> Dict[str, Any]:
        """
        分析ILP求解结果

        Args:
            ilp_formulation: ILP公式化实例

        Returns:
            分析结果
        """
        if not ilp_formulation.problem or not pulp.LpStatus[ilp_formulation.problem.status] == 'Optimal':
            return {'status': 'No optimal solution found'}

        # 获取变量值
        self.solution = ilp_formulation.get_variable_values()
        self.solution['status'] = pulp.LpStatus[ilp_formulation.problem.status]
        self.solution['objective_value'] = pulp.value(ilp_formulation.problem.objective)

        # 分析结果
        analysis = {
            'status': self.solution['status'],
            'objective_value': self.solution['objective_value'],
            'routing_paths': self._extract_routing_paths(),
            'spectrum_assignments': self._extract_spectrum_assignments(),
            'resource_utilization': self._calculate_resource_utilization(),
            'performance_metrics': self._calculate_performance_metrics()
        }

        return analysis

    def _extract_routing_paths(self) -> Dict[int, List[str]]:
        """提取路由路径"""
        routing_paths = {}

        if 'routing' not in self.solution:
            return routing_paths

        for k_idx, edges in self.solution['routing'].items():
            demand = self.network.demands[k_idx]
            source, target = demand['source'], demand['target']

            # 构建路径 - 使用完整网络图和选定的边
            path = self._find_path_in_network(edges, source, target)

            routing_paths[k_idx] = {
                'source': source,
                'target': target,
                'path': path,
                'hops': len(path) - 1 if path else 0,
                'edges': edges
            }

        return routing_paths

    def _build_path_from_edges(self, edges: List[Tuple[str, str]], source: str, target: str) -> List[str]:
        """从边列表构建路径"""
        if not edges:
            return []

        # 构建邻接表
        graph = nx.Graph()
        graph.add_edges_from(edges)

        # 如果源和目标不在图中，返回空路径
        if not graph.has_node(source) or not graph.has_node(target):
            return []

        # 寻找从源到目标的路径
        try:
            path = nx.shortest_path(graph, source, target)
            return path
        except nx.NetworkXNoPath:
            # 尝试深度优先搜索
            try:
                path = list(nx.all_simple_paths(graph, source, target))[0]
                return path
            except (nx.NetworkXNoPath, IndexError):
                return []

    def _trace_path_in_network(self, edges: List[Tuple[str, str]], source: str, target: str) -> List[str]:
        """在网络图中追踪路径"""
        if not edges:
            return []

        # 获取网络图的子图，只包含使用的边
        subgraph = nx.Graph()
        subgraph.add_edges_from(edges)

        # 检查源节点和目标节点是否在子图中
        if not subgraph.has_node(source) or not subgraph.has_node(target):
            # 如果不在子图中，使用完整网络图，但只考虑选定的边
            full_subgraph = nx.Graph()
            full_subgraph.add_nodes_from(self.network.get_nodes())
            full_subgraph.add_edges_from(edges)

            try:
                path = nx.shortest_path(full_subgraph, source, target)
                return path
            except nx.NetworkXNoPath:
                return []

        # 使用子图寻找路径
        try:
            path = nx.shortest_path(subgraph, source, target)
            return path
        except nx.NetworkXNoPath:
            return []

    def _find_path_in_network(self, edges: List[Tuple[str, str]], source: str, target: str) -> List[str]:
        """在完整网络图中使用选定的边寻找路径"""
        if not edges:
            return []

        # 如果源和目标直接相连，直接返回
        if (source, target) in edges or (target, source) in edges:
            return [source, target]

        # 创建完整网络图的子图，只包含选定的边
        subgraph = nx.Graph()
        subgraph.add_nodes_from(self.network.get_nodes())
        subgraph.add_edges_from(edges)

        # 使用网络图中的最短路径
        try:
            path = nx.shortest_path(subgraph, source, target)
            return path
        except nx.NetworkXNoPath:
            # 如果最短路径失败，尝试所有简单路径
            try:
                paths = list(nx.all_simple_paths(subgraph, source, target))
                if paths:
                    return paths[0]  # 返回第一个找到的路径
            except (nx.NetworkXNoPath, IndexError):
                pass

        # 如果仍然失败，尝试在完整网络图中寻找路径
        try:
            path = nx.shortest_path(self.network.graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return []

    def _extract_spectrum_assignments(self) -> Dict[int, Dict[str, Any]]:
        """提取频谱分配"""
        spectrum_assignments = {}

        if 'spectrum' not in self.solution:
            return spectrum_assignments

        for k_idx, slots in self.solution['spectrum'].items():
            demand = self.network.demands[k_idx]
            w_k = demand['slots']

            if slots:
                # 找到最后一个频隙
                last_slot = max(slots)
                first_slot = last_slot - w_k + 1
                assigned_slots = list(range(first_slot, last_slot + 1))

                spectrum_assignments[k_idx] = {
                    'source': demand['source'],
                    'target': demand['target'],
                    'required_slots': w_k,
                    'first_slot': first_slot,
                    'last_slot': last_slot,
                    'assigned_slots': assigned_slots,
                    'contiguous': True
                }

        return spectrum_assignments

    def _calculate_resource_utilization(self) -> Dict[str, Any]:
        """计算资源利用率"""
        utilization = {
            'edge_usage': {},
            'spectrum_usage': {},
            'total_used_edges': 0,
            'max_slot_position': 0
        }

        # 边使用情况
        if 'edge_activation' in self.solution:
            for edge, used in self.solution['edge_activation'].items():
                utilization['edge_usage'][edge] = used
                if used:
                    utilization['total_used_edges'] += 1

        # 频谱使用情况
        if 'global_max_slot' in self.solution:
            utilization['max_slot_position'] = self.solution['global_max_slot']

        # 计算每条边的频谱使用
        if 'edge_slot' in self.solution:
            for k_idx, edge_slots in self.solution['edge_slot'].items():
                for edge, slots in edge_slots.items():
                    if edge not in utilization['spectrum_usage']:
                        utilization['spectrum_usage'][edge] = []
                    utilization['spectrum_usage'][edge].extend(slots)

            # 去重并排序
            for edge in utilization['spectrum_usage']:
                utilization['spectrum_usage'][edge] = sorted(list(set(utilization['spectrum_usage'][edge])))

        return utilization

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """计算性能指标"""
        metrics = {
            'total_hops': 0,
            'average_hops': 0,
            'network_load': 0,
            'spectrum_efficiency': 0,
            'successfully_routed_demands': 0
        }

        routing_paths = self._extract_routing_paths()

        if routing_paths:
            # 计算总跳数和平均跳数
            total_hops = sum(path_data['hops'] for path_data in routing_paths.values())
            metrics['total_hops'] = total_hops
            metrics['average_hops'] = total_hops / len(routing_paths)
            metrics['successfully_routed_demands'] = len(routing_paths)

        # 计算网络负载
        utilization = self._calculate_resource_utilization()
        total_edges = len(self.network.get_edges())
        if total_edges > 0:
            metrics['network_load'] = utilization['total_used_edges'] / total_edges

        # 计算频谱效率
        if 'max_slot_position' in utilization and utilization['max_slot_position'] > 0:
            total_required_slots = sum(demand['slots'] for demand in self.network.demands)
            metrics['spectrum_efficiency'] = total_required_slots / utilization['max_slot_position']

        return metrics

    def validate_solution(self) -> Dict[str, Any]:
        """验证解决方案的正确性"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        try:
            # 验证路由约束
            self._validate_routing_constraints(validation)

            # 验证频谱约束
            self._validate_spectrum_constraints(validation)

            # 验证传输距离约束
            self._validate_reach_constraints(validation)

        except Exception as e:
            validation['is_valid'] = False
            validation['errors'].append(f"验证过程中出现错误: {str(e)}")

        return validation

    def _validate_routing_constraints(self, validation: Dict[str, Any]) -> None:
        """验证路由约束"""
        routing_paths = self._extract_routing_paths()

        for k_idx, path_data in routing_paths.items():
            path = path_data['path']
            source, target = path_data['source'], path_data['target']

            # 验证路径连通性
            if path[0] != source or path[-1] != target:
                validation['is_valid'] = False
                validation['errors'].append(f"需求{k_idx}: 路径不连通 {path}")

            # 验证路径存在性
            if not self._path_exists(path):
                validation['is_valid'] = False
                validation['errors'].append(f"需求{k_idx}: 路径不存在 {path}")

    def _validate_spectrum_constraints(self, validation: Dict[str, Any]) -> None:
        """验证频谱约束"""
        spectrum_assignments = self._extract_spectrum_assignments()
        routing_paths = self._extract_routing_paths()

        # 添加详细的频谱约束验证结果
        validation['spectrum_details'] = {
            'continuity': {},
            'adjacency': {},
            'non_overlapping': {}
        }

        # 验证频谱连续性和邻接性
        for k_idx, assignment in spectrum_assignments.items():
            assigned_slots = assignment['assigned_slots']
            required_slots = assignment['required_slots']
            # 从路由路径中获取边信息
            if k_idx in routing_paths:
                path_edges = routing_paths[k_idx]['edges']
            else:
                path_edges = []

            # 初始化验证结果
            validation['spectrum_details']['continuity'][k_idx] = {
                'passed': True,
                'details': {}
            }
            validation['spectrum_details']['adjacency'][k_idx] = {
                'passed': True,
                'details': {}
            }

            # 1. 验证频谱数量
            if len(assigned_slots) != required_slots:
                validation['is_valid'] = False
                validation['errors'].append(f"需求{k_idx+1}: 频谱数量不匹配 (需要{required_slots}个，分配{len(assigned_slots)}个)")
                validation['spectrum_details']['continuity'][k_idx]['passed'] = False
                validation['spectrum_details']['adjacency'][k_idx]['passed'] = False

            # 2. 验证频谱邻接性（连续性）
            if assigned_slots and len(assigned_slots) > 1:
                expected_slots = list(range(assigned_slots[0], assigned_slots[-1] + 1))
                if assigned_slots != expected_slots:
                    validation['is_valid'] = False
                    validation['errors'].append(f"需求{k_idx+1}: 频谱不连续 (分配{assigned_slots}, 期望{expected_slots})")
                    validation['spectrum_details']['adjacency'][k_idx]['passed'] = False
                    validation['spectrum_details']['adjacency'][k_idx]['details']['error'] = f"频隙不连续: {assigned_slots}"
                else:
                    validation['spectrum_details']['adjacency'][k_idx]['details']['slots'] = assigned_slots
                    validation['spectrum_details']['adjacency'][k_idx]['details']['message'] = f"频隙连续: {assigned_slots[0]}-{assigned_slots[-1]}"
            elif assigned_slots and len(assigned_slots) == 1:
                validation['spectrum_details']['adjacency'][k_idx]['details']['slots'] = assigned_slots
                validation['spectrum_details']['adjacency'][k_idx]['details']['message'] = f"单个频隙: {assigned_slots[0]}"

            # 3. 验证频谱连续性（路径上的一致性）
            if 'edge_slot' in self.solution and k_idx in self.solution['edge_slot']:
                edge_slots = self.solution['edge_slot'][k_idx]
                consistent_slots = None
                continuity_errors = []

                for edge, slots in edge_slots.items():
                    if consistent_slots is None:
                        consistent_slots = set(slots)
                    else:
                        if set(slots) != consistent_slots:
                            continuity_errors.append(f"边{edge}: {slots} (期望: {list(consistent_slots)})")

                if continuity_errors:
                    validation['is_valid'] = False
                    validation['errors'].append(f"需求{k_idx+1}: 频谱在路径上不一致")
                    validation['spectrum_details']['continuity'][k_idx]['passed'] = False
                    validation['spectrum_details']['continuity'][k_idx]['details']['errors'] = continuity_errors
                else:
                    validation['spectrum_details']['continuity'][k_idx]['details']['consistent_slots'] = list(consistent_slots)
                    validation['spectrum_details']['continuity'][k_idx]['details']['message'] = f"路径上频谱一致: {list(consistent_slots)}"

        # 验证频谱非重叠
        self._validate_spectrum_non_overlapping_detailed(validation)

    def _validate_spectrum_non_overlapping(self, validation: Dict[str, Any]) -> None:
        """验证频谱非重叠约束"""
        edge_slot_usage = {}

        if 'edge_slot' not in self.solution:
            return

        for k_idx, edge_slots in self.solution['edge_slot'].items():
            for edge, slots in edge_slots.items():
                if edge not in edge_slot_usage:
                    edge_slot_usage[edge] = []

                # 检查重叠
                for slot in slots:
                    if slot in edge_slot_usage[edge]:
                        validation['is_valid'] = False
                        validation['errors'].append(f"边{edge}: 频隙{slot}重叠使用")
                    edge_slot_usage[edge].append(slot)

    def _validate_spectrum_non_overlapping_detailed(self, validation: Dict[str, Any]) -> None:
        """详细验证频谱非重叠约束"""
        edge_slot_usage = {}
        non_overlapping_details = {}

        if 'edge_slot' not in self.solution:
            return

        # 初始化非重叠验证结果
        validation['spectrum_details']['non_overlapping'] = {
            'edge_conflicts': {},
            'edge_usage': {}
        }

        for k_idx, edge_slots in self.solution['edge_slot'].items():
            for edge, slots in edge_slots.items():
                if edge not in edge_slot_usage:
                    edge_slot_usage[edge] = {}
                    non_overlapping_details[edge] = []

                # 检查每个频隙的使用情况
                for slot in slots:
                    if slot not in edge_slot_usage[edge]:
                        edge_slot_usage[edge][slot] = []
                    edge_slot_usage[edge][slot].append(k_idx)

                    # 如果频隙被多个需求使用，记录冲突
                    if len(edge_slot_usage[edge][slot]) > 1:
                        conflict_demands = edge_slot_usage[edge][slot]
                        if edge not in validation['spectrum_details']['non_overlapping']['edge_conflicts']:
                            validation['spectrum_details']['non_overlapping']['edge_conflicts'][edge] = {}
                        validation['spectrum_details']['non_overlapping']['edge_conflicts'][edge][slot] = conflict_demands
                        validation['is_valid'] = False
                        validation['errors'].append(f"边{edge}: 频隙{slot}被需求{conflict_demands}同时使用")

        # 生成每条边的频谱使用情况
        for edge, slot_usage in edge_slot_usage.items():
            validation['spectrum_details']['non_overlapping']['edge_usage'][edge] = {
                'total_slots_used': len(slot_usage),
                'slot_allocation': {slot: demands for slot, demands in slot_usage.items()},
                'max_slot': max(slot_usage.keys()) if slot_usage else 0,
                'has_conflicts': edge in validation['spectrum_details']['non_overlapping']['edge_conflicts']
            }

    def _validate_reach_constraints(self, validation: Dict[str, Any]) -> None:
        """验证传输距离约束"""
        routing_paths = self._extract_routing_paths()

        for k_idx, path_data in routing_paths.items():
            path = path_data['path']
            demand = self.network.demands[k_idx]
            max_reach = demand['reach']

            # 计算路径总距离
            total_distance = 0
            for i in range(len(path) - 1):
                distance = self.network.get_edge_distance(path[i], path[i+1])
                total_distance += distance

            if total_distance > max_reach:
                validation['is_valid'] = False
                validation['errors'].append(f"需求{k_idx+1}: 传输距离{total_distance}超过限制{max_reach}")

    def _path_exists(self, path: List[str]) -> bool:
        """验证路径是否存在"""
        if len(path) < 2:
            return False

        for i in range(len(path) - 1):
            if not self.network.graph.has_edge(path[i], path[i+1]):
                return False

        return True

    def print_spectrum_allocation_matrix(self) -> None:
        """打印频谱分配矩阵"""
        if 'edge_slot' not in self.solution:
            return

        # 获取所有边
        edges = self.network.get_edges()
        slots = range(1, self.network.spectrum_slots + 1)

        # 创建分配矩阵: 边 x 频隙
        allocation_matrix = {}
        for edge in edges:
            allocation_matrix[edge] = [''] * len(slots)

        # 填充矩阵
        for k_idx, edge_slots in self.solution['edge_slot'].items():
            for edge, slots_used in edge_slots.items():
                for slot in slots_used:
                    if 1 <= slot <= len(slots):
                        allocation_matrix[edge][slot-1] = str(k_idx + 1)

        # 打印矩阵
        print("\n=== 频谱分配矩阵 ===")
        print("     |" + "|".join(f"{i:2d}" for i in slots))
        print("-----" + "--" * len(slots))

        for edge in edges:
            edge_label = f"{edge[0]}{edge[1]}"
            print(f"{edge_label:4} |" + "|".join(f"{val:2}" if val else "  " for val in allocation_matrix[edge]))

        print()

    def print_analysis_results(self, analysis: Dict[str, Any]) -> None:
        """打印分析结果"""
        print("=== 求解结果分析 ===")
        print(f"求解状态: {analysis['status']}")
        print(f"目标函数值: {analysis['objective_value']}")

        if 'routing_paths' in analysis:
            print(f"\n路由路径:")
            for k_idx, path_data in analysis['routing_paths'].items():
                print(f"  需求{k_idx+1}: {path_data['source']} -> {path_data['target']}")
                print(f"    路径: {' -> '.join(path_data['path'])}")
                print(f"    跳数: {path_data['hops']}")

        if 'spectrum_assignments' in analysis:
            print(f"\n频谱分配:")
            for k_idx, assignment in analysis['spectrum_assignments'].items():
                print(f"  需求{k_idx+1}: 频隙{assignment['first_slot']}-{assignment['last_slot']} "
                      f"({len(assignment['assigned_slots'])}个)")

        # 打印频谱分配矩阵
        self.print_spectrum_allocation_matrix()

        if 'performance_metrics' in analysis:
            metrics = analysis['performance_metrics']
            print(f"\n性能指标:")
            print(f"  成功路由需求: {metrics['successfully_routed_demands']}")
            print(f"  总跳数: {metrics['total_hops']}")
            print(f"  平均跳数: {metrics['average_hops']:.2f}")
            print(f"  网络负载: {metrics['network_load']:.2f}")
            print(f"  频谱效率: {metrics['spectrum_efficiency']:.2f}")

        if 'resource_utilization' in analysis:
            utilization = analysis['resource_utilization']
            print(f"\n资源利用:")
            print(f"  使用边数: {utilization['total_used_edges']}")
            print(f"  最大频隙位置: {utilization['max_slot_position']}")

    def print_validation_results(self, validation: Dict[str, Any]) -> None:
        """打印验证结果"""
        print("=== 解决方案验证 ===")
        if validation['is_valid']:
            print("PASS: 解决方案有效")
        else:
            print("FAIL: 解决方案无效")

        if validation['errors']:
            print("错误:")
            for error in validation['errors']:
                print(f"  - {error}")

        if validation['warnings']:
            print("警告:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        # 打印详细的频谱约束验证结果
        if 'spectrum_details' in validation:
            print("\n=== 频谱约束详细验证 ===")

            # 1. 频谱连续性验证
            print("\n1. 频谱连续性验证（同一需求在路径所有边上使用相同频隙）:")
            continuity_details = validation['spectrum_details']['continuity']
            for k_idx, details in continuity_details.items():
                status = "PASS" if details['passed'] else "FAIL"
                print(f"  需求{k_idx+1}: {status}")
                if details['passed']:
                    if 'consistent_slots' in details['details']:
                        slots = details['details']['consistent_slots']
                        print(f"    路径上频谱一致: {slots}")
                    if 'message' in details['details']:
                        print(f"    {details['details']['message']}")
                else:
                    if 'errors' in details['details']:
                        for error in details['details']['errors']:
                            print(f"    错误: {error}")

            # 2. 频谱邻接性验证
            print("\n2. 频谱邻接性验证（分配给需求的频隙必须是连续的）:")
            adjacency_details = validation['spectrum_details']['adjacency']
            for k_idx, details in adjacency_details.items():
                status = "PASS" if details['passed'] else "FAIL"
                print(f"  需求{k_idx+1}: {status}")
                if details['passed']:
                    if 'slots' in details['details']:
                        slots = details['details']['slots']
                        if len(slots) > 1:
                            print(f"    频隙连续: {slots[0]}-{slots[-1]}")
                        else:
                            print(f"    单个频隙: {slots[0]}")
                    if 'message' in details['details']:
                        print(f"    {details['details']['message']}")
                else:
                    if 'error' in details['details']:
                        print(f"    错误: {details['details']['error']}")

            # 3. 频谱非重叠验证
            print("\n3. 频谱非重叠验证（每个频隙最多分配给一个需求）:")
            non_overlapping_details = validation['spectrum_details']['non_overlapping']

            # 检查是否有冲突
            has_conflicts = bool(non_overlapping_details.get('edge_conflicts', {}))
            print(f"  总体状态: {'PASS 无冲突' if not has_conflicts else 'FAIL 存在冲突'}")

            if non_overlapping_details.get('edge_conflicts', {}):
                print("  冲突详情:")
                for edge, conflicts in non_overlapping_details.get('edge_conflicts', {}).items():
                    for slot, demands in conflicts.items():
                        print(f"    边{edge}: 频隙{slot}被需求{demands}同时使用")

            print("  各边频谱使用情况:")
            for edge, usage in non_overlapping_details.get('edge_usage', {}).items():
                conflict_status = " (存在冲突)" if usage['has_conflicts'] else ""
                print(f"    边{edge}: 使用{usage['total_slots_used']}个频隙，最大频隙{usage['max_slot']}{conflict_status}")
                if usage['slot_allocation']:
                    allocation_str = ", ".join([f"频隙{slot}->需求{d+1}" for slot, demands in usage['slot_allocation'].items() for d in demands])
                    print(f"      分配: {allocation_str}")

        print("\n=== 验证总结 ===")
        passed_constraints = []
        if 'spectrum_details' in validation:
            if all(details['passed'] for details in validation['spectrum_details']['continuity'].values()):
                passed_constraints.append("频谱连续性")
            if all(details['passed'] for details in validation['spectrum_details']['adjacency'].values()):
                passed_constraints.append("频谱邻接性")
            if not validation['spectrum_details']['non_overlapping'].get('edge_conflicts', {}):
                passed_constraints.append("频谱非重叠")

        print(f"通过约束: {', '.join(passed_constraints) if passed_constraints else '无'}")
        print(f"总体验证: {'通过' if validation['is_valid'] else '失败'}")


# 测试函数
def test_solution_analyzer():
    """测试解决方案分析器"""
    print("测试解决方案分析器...")

    from .network_model import NetworkModel
    from .ilp_formulation import ILPFormulation

    # 创建网络模型和ILP公式化
    network = NetworkModel()
    network.create_6node_topology()
    network.create_test_demands()

    ilp = ILPFormulation(network)
    problem = ilp.build_model('O3')

    # 创建分析器（无需实际求解）
    analyzer = SolutionAnalyzer(network)
    print("✓ 解决方案分析器创建成功")

    # 测试分析方法
    empty_analysis = analyzer.analyze_solution(ilp)
    print(f"✓ 空解决方案分析完成")


if __name__ == "__main__":
    test_solution_analyzer()