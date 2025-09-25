#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILP RSA求解器主程序
实现论文中的路由与频谱分配ILP模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.network_model import NetworkModel
from core.ilp_formulation import ILPFormulation
from core.solution_analyzer import SolutionAnalyzer
import pulp
import time


def main():
    """主程序"""
    print("=== ILP RSA求解器 ===")
    print("基于论文: A novel integer linear programming model for routing and spectrum assignment in optical networks")
    print()

    # 1. 创建网络模型
    print("1. 创建网络模型...")
    network = NetworkModel()
    network.create_6node_topology()
    network.create_test_demands()
    network.print_network_info()
    print()

    # 2. 构建ILP模型
    print("2. 构建ILP模型...")
    ilp = ILPFormulation(network)

    # 选择优化目标 (MULTI: 多目标优先级 O3 > O1 > O2)
    objective_type = 'O3'
    print(f"   优化目标: {objective_type} (多目标优先级: O3 > O1 > O2)")

    start_time = time.time()
    problem = ilp.build_model(objective_type)
    build_time = time.time() - start_time

    print(f"   模型构建完成，用时 {build_time:.2f} 秒")
    print(f"   变量数量: {len(problem.variables())}")
    print(f"   约束数量: {len(problem.constraints)}")
    print()

    # 3. 求解ILP模型
    print("3. 求解ILP模型...")
    print(f"   使用求解器: {pulp.listSolvers(onlyAvailable=True)}")

    start_time = time.time()
    # 优化求解器配置
    solver = pulp.PULP_CBC_CMD(
        msg=True,
        timeLimit=300,        # 5分钟时间限制
        gapRel=0.01,          # 相对gap 1%
        gapAbs=0.9,           # 绝对gap 0.1
        cuts=True,            # 启用割平面
        presolve=True,        # 启用预处理
        strong=False,         # 禁用强分支（用于小规模问题）
        options=[
            'cuts on',
            'presolve on',
            'round integer variables',
            'heuristics on',
            'mixed integer rounding cuts on',
            'gomory cuts on',
            'clique cuts on',
            'flow cover cuts on'
        ]
    )
    status = problem.solve(solver)
    solve_time = time.time() - start_time

    print(f"   求解完成，用时 {solve_time:.2f} 秒")
    print(f"   求解状态: {pulp.LpStatus[status]}")
    print()

    # 4. 分析结果
    print("4. 分析求解结果...")
    analyzer = SolutionAnalyzer(network)

    if pulp.LpStatus[status] == 'Optimal':
        # 分析解决方案
        analysis = analyzer.analyze_solution(ilp)
        analyzer.print_analysis_results(analysis)

        # 验证解决方案
        validation = analyzer.validate_solution()
        analyzer.print_validation_results(validation)

        print(f"\n=== 总结 ===")
        print(f"总用时: {build_time + solve_time:.2f} 秒")
        print(f"目标函数值: {analysis['objective_value']}")
        print(f"成功路由需求: {analysis['performance_metrics']['successfully_routed_demands']}/{len(network.demands)}")

        if validation['is_valid']:
            print("PASS: 解决方案验证通过")
        else:
            print("FAIL: 解决方案验证失败")

    else:
        print(f"求解失败，状态: {pulp.LpStatus[status]}")

    # 5. 保存结果
    print(f"\n5. 保存结果...")
    network.save_to_json("E:/PythonProject/ILP/log/solved_network.json")

    print("程序执行完成")


def test_different_objectives():
    """测试不同优化目标"""
    print("=== 测试不同优化目标 ===")

    network = NetworkModel()
    network.create_6node_topology()
    network.create_test_demands()

    objectives = ['O1', 'O2', 'O3']
    results = {}

    for obj in objectives:
        print(f"\n测试目标 {obj}:")
        ilp = ILPFormulation(network)
        problem = ilp.build_model(obj)

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
        status = problem.solve(solver)

        if pulp.LpStatus[status] == 'Optimal':
            results[obj] = {
                'status': pulp.LpStatus[status],
                'objective_value': pulp.value(problem.objective),
                'variables': len(problem.variables()),
                'constraints': len(problem.constraints)
            }
            print(f"  OK 目标函数值: {results[obj]['objective_value']}")
        else:
            results[obj] = {
                'status': pulp.LpStatus[status],
                'objective_value': None,
                'variables': len(problem.variables()),
                'constraints': len(problem.constraints)
            }
            print(f"  FAIL 求解失败: {results[obj]['status']}")

    print(f"\n=== 不同目标对比 ===")
    for obj, result in results.items():
        print(f"{obj}: {result['status']}")
        if result['objective_value'] is not None:
            print(f"    目标值: {result['objective_value']}")


if __name__ == "__main__":
    # 运行主程序
    main()

    # 可选：测试不同目标
    # test_different_objectives()