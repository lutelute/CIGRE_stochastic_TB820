#!/usr/bin/env python3
"""
高度最適化ノートブック (02_advanced_optimization.ipynb) の主要部分を実行
"""

import sys
import os
sys.path.append('02-python-implementation/src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, optimize
import warnings
warnings.filterwarnings('ignore')

print("🚀 CIGRE TB820 高度最適化ノートブック実行開始")
print("="*60)

# ============================================================================
# 1. CVaR最適化の実装
# ============================================================================
print("\n💎 1. CVaR最適化の実装")
print("-" * 40)

try:
    import cvxpy as cp
    print("✓ CVXPY利用可能 - CVaR最適化を実行")
    
    class CVaROptimizer:
        """CVaR最適化クラス"""
        
        def __init__(self, wind_scenarios, probabilities, demand=80, 
                     conv_cost=60, emergency_cost=200, confidence_level=0.95):
            self.wind_scenarios = wind_scenarios
            self.probabilities = probabilities
            self.demand = demand
            self.conv_cost = conv_cost
            self.emergency_cost = emergency_cost
            self.confidence_level = confidence_level
            
        def solve_cvar_optimization(self):
            """CVXPYを使用したCVaR最適化"""
            n_scenarios = len(self.wind_scenarios)
            
            # 決定変数
            x = cp.Variable(nonneg=True, name="conventional_gen")
            eta = cp.Variable(name="var_estimate")
            z = cp.Variable(n_scenarios, nonneg=True, name="excess_loss")
            
            # 各シナリオでのコスト
            scenario_costs = []
            for i, wind in enumerate(self.wind_scenarios):
                shortage = cp.maximum(0, self.demand - x - wind)
                cost = self.conv_cost * x + self.emergency_cost * shortage
                scenario_costs.append(cost)
            
            # CVaR制約
            constraints = []
            for i in range(n_scenarios):
                constraints.append(z[i] >= scenario_costs[i] - eta)
            
            constraints.append(x <= 100)  # 発電容量制約
            
            # CVaR目的関数
            alpha = 1 - self.confidence_level
            cvar_objective = eta + (1/alpha) * cp.sum(cp.multiply(self.probabilities, z))
            
            # 問題定義と求解
            problem = cp.Problem(cp.Minimize(cvar_objective), constraints)
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                return {
                    'optimal_x': x.value,
                    'var_estimate': eta.value,
                    'cvar_value': cvar_objective.value,
                    'scenario_costs': [cost.value for cost in scenario_costs],
                    'status': 'optimal'
                }
            else:
                return {'status': 'failed', 'problem_status': problem.status}
        
        def solve_expected_value_optimization(self):
            """期待値最適化（比較用）"""
            x = cp.Variable(nonneg=True)
            
            expected_cost = 0
            for i, wind in enumerate(self.wind_scenarios):
                shortage = cp.maximum(0, self.demand - x - wind)
                cost = self.conv_cost * x + self.emergency_cost * shortage
                expected_cost += self.probabilities[i] * cost
            
            constraints = [x <= 100]
            problem = cp.Problem(cp.Minimize(expected_cost), constraints)
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                return {
                    'optimal_x': x.value,
                    'expected_cost': expected_cost.value,
                    'status': 'optimal'
                }
            else:
                return {'status': 'failed'}

    # シナリオ生成
    np.random.seed(42)
    n_scenarios = 100
    wind_mean, wind_std = 50, 10

    wind_scenarios = np.random.normal(wind_mean, wind_std, n_scenarios)
    probabilities = np.ones(n_scenarios) / n_scenarios

    print(f"生成されたシナリオ数: {n_scenarios}")
    print(f"風力出力範囲: [{wind_scenarios.min():.1f}, {wind_scenarios.max():.1f}] MW")

    # CVaR最適化の実行
    confidence_levels = [0.90, 0.95, 0.99]
    results = []

    print(f"\n=== CVaR最適化結果 ===")

    for conf_level in confidence_levels:
        optimizer = CVaROptimizer(wind_scenarios, probabilities, confidence_level=conf_level)
        
        # CVaR最適化
        cvar_result = optimizer.solve_cvar_optimization()
        
        if cvar_result['status'] == 'optimal':
            results.append({
                'confidence_level': conf_level,
                'method': 'CVaR',
                'optimal_x': cvar_result['optimal_x'],
                'objective_value': cvar_result['cvar_value'],
                'var_estimate': cvar_result['var_estimate']
            })
            
            print(f"信頼水準 {conf_level:.0%}:")
            print(f"  最適従来発電量: {cvar_result['optimal_x']:.2f} MW")
            print(f"  CVaR値: ${cvar_result['cvar_value']:.2f}")
            print(f"  VaR推定値: ${cvar_result['var_estimate']:.2f}")

    # 期待値最適化（比較用）
    optimizer_ev = CVaROptimizer(wind_scenarios, probabilities)
    ev_result = optimizer_ev.solve_expected_value_optimization()

    if ev_result['status'] == 'optimal':
        results.append({
            'confidence_level': 'Expected',
            'method': 'EV',
            'optimal_x': ev_result['optimal_x'],
            'objective_value': ev_result['expected_cost']
        })
        
        print(f"\n期待値最適化:")
        print(f"  最適従来発電量: {ev_result['optimal_x']:.2f} MW")
        print(f"  期待コスト: ${ev_result['expected_cost']:.2f}")

    # 結果をDataFrameで整理
    results_df = pd.DataFrame(results)
    print(f"\n=== 結果比較表 ===")
    print(results_df.round(2))

except ImportError:
    print("⚠️  CVXPY未インストール - CVaR最適化をスキップ")
    print("   sudo pip install cvxpy で追加インストール可能")
    results_df = None

# ============================================================================
# 2. 二段階確率計画法
# ============================================================================
print(f"\n🏗️  2. 二段階確率計画法")
print("-" * 40)

try:
    class TwoStageStochasticPlanning:
        """二段階確率計画法クラス"""
        
        def __init__(self, scenarios_data):
            self.scenarios = scenarios_data
            self.n_scenarios = len(scenarios_data)
            
            # 発電機オプション (容量MW, 投資コスト$/MW, 運用コスト$/MWh)
            self.generator_options = {
                'coal': {'capacity': 100, 'investment_cost': 1000, 'operating_cost': 40},
                'gas': {'capacity': 80, 'investment_cost': 600, 'operating_cost': 80},
                'renewable': {'capacity': 120, 'investment_cost': 800, 'operating_cost': 10}
            }
            
        def solve_two_stage_problem(self):
            """二段階確率計画問題を解く"""
            # 第一段階変数（投資決定）
            invest_vars = {}
            for gen_type in self.generator_options.keys():
                invest_vars[gen_type] = cp.Variable(nonneg=True, name=f"invest_{gen_type}")
            
            # 第二段階変数（運用決定）
            operating_vars = {}
            for s in range(self.n_scenarios):
                operating_vars[s] = {}
                for gen_type in self.generator_options.keys():
                    operating_vars[s][gen_type] = cp.Variable(
                        nonneg=True, name=f"operate_{gen_type}_s{s}"
                    )
                operating_vars[s]['shortage'] = cp.Variable(
                    nonneg=True, name=f"shortage_s{s}"
                )
            
            # 制約条件
            constraints = []
            
            # 投資制約（予算制約）
            total_investment = sum(
                invest_vars[gen_type] * self.generator_options[gen_type]['investment_cost']
                for gen_type in self.generator_options.keys()
            )
            constraints.append(total_investment <= 100000)  # 予算10万ドル
            
            # 各シナリオの運用制約
            for s in range(self.n_scenarios):
                scenario = self.scenarios[s]
                demand = scenario['demand']
                wind_output = scenario['wind']
                
                # 需給バランス
                total_generation = sum(
                    operating_vars[s][gen_type] 
                    for gen_type in self.generator_options.keys()
                )
                constraints.append(
                    total_generation + wind_output + operating_vars[s]['shortage'] >= demand
                )
                
                # 発電容量制約
                for gen_type in self.generator_options.keys():
                    max_capacity = invest_vars[gen_type] * self.generator_options[gen_type]['capacity']
                    constraints.append(operating_vars[s][gen_type] <= max_capacity)
            
            # 目的関数（投資コスト + 期待運用コスト）
            investment_cost = total_investment
            
            expected_operating_cost = 0
            for s in range(self.n_scenarios):
                scenario_prob = self.scenarios[s]['probability']
                
                scenario_cost = 0
                for gen_type in self.generator_options.keys():
                    op_cost = self.generator_options[gen_type]['operating_cost']
                    scenario_cost += op_cost * operating_vars[s][gen_type]
                
                # 不足電力ペナルティ
                scenario_cost += 500 * operating_vars[s]['shortage']  # $/MWh
                
                expected_operating_cost += scenario_prob * scenario_cost
            
            total_cost = investment_cost + expected_operating_cost
            
            # 問題求解
            problem = cp.Problem(cp.Minimize(total_cost), constraints)
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                # 結果の整理
                investment_results = {}
                for gen_type in self.generator_options.keys():
                    investment_results[gen_type] = invest_vars[gen_type].value
                
                operating_results = []
                for s in range(self.n_scenarios):
                    scenario_result = {'scenario': s}
                    for gen_type in self.generator_options.keys():
                        scenario_result[gen_type] = operating_vars[s][gen_type].value
                    scenario_result['shortage'] = operating_vars[s]['shortage'].value
                    operating_results.append(scenario_result)
                
                return {
                    'status': 'optimal',
                    'total_cost': total_cost.value,
                    'investment_cost': investment_cost.value,
                    'expected_operating_cost': expected_operating_cost.value,
                    'investments': investment_results,
                    'operations': operating_results
                }
            else:
                return {'status': 'failed', 'problem_status': problem.status}

    # シナリオデータの生成
    np.random.seed(123)
    n_scenarios_2stage = 20

    scenarios_data = []
    for s in range(n_scenarios_2stage):
        scenarios_data.append({
            'demand': np.random.normal(100, 15),  # MW
            'wind': np.random.normal(30, 8),      # MW
            'probability': 1.0 / n_scenarios_2stage
        })

    print(f"生成したシナリオ数: {n_scenarios_2stage}")
    demands = [s['demand'] for s in scenarios_data]
    winds = [s['wind'] for s in scenarios_data]
    print(f"需要範囲: [{min(demands):.1f}, {max(demands):.1f}] MW")
    print(f"風力範囲: [{min(winds):.1f}, {max(winds):.1f}] MW")

    # 二段階確率計画法の実行
    two_stage_planner = TwoStageStochasticPlanning(scenarios_data)
    result = two_stage_planner.solve_two_stage_problem()

    if result['status'] == 'optimal':
        print(f"\n=== 二段階確率計画法結果 ===")
        print(f"総コスト: ${result['total_cost']:,.0f}")
        print(f"投資コスト: ${result['investment_cost']:,.0f}")
        print(f"期待運用コスト: ${result['expected_operating_cost']:,.0f}")
        
        print(f"\n=== 最適投資戦略 ===")
        investments = result['investments']
        for gen_type, units in investments.items():
            if units > 0.01:  # 小さな値は除外
                capacity = units * two_stage_planner.generator_options[gen_type]['capacity']
                cost = units * two_stage_planner.generator_options[gen_type]['investment_cost']
                print(f"{gen_type.title()}: {units:.2f}ユニット ({capacity:.0f}MW, ${cost:,.0f})")
        
        # 運用結果の統計
        operations_df = pd.DataFrame(result['operations'])
        
        print(f"\n=== 運用統計 ===")
        for gen_type in ['coal', 'gas', 'renewable']:
            if gen_type in operations_df.columns:
                avg_output = operations_df[gen_type].mean()
                max_output = operations_df[gen_type].max()
                print(f"{gen_type.title()}: 平均 {avg_output:.1f}MW, 最大 {max_output:.1f}MW")
        
        avg_shortage = operations_df['shortage'].mean()
        max_shortage = operations_df['shortage'].max()
        shortage_freq = (operations_df['shortage'] > 0.1).mean()
        print(f"不足電力: 平均 {avg_shortage:.1f}MW, 最大 {max_shortage:.1f}MW, 頻度 {shortage_freq:.1%}")

    else:
        print(f"✗ 二段階最適化失敗: {result.get('problem_status', 'Unknown error')}")

except ImportError:
    print("⚠️  CVXPY未インストール - 二段階確率計画法をスキップ")
    print("   sudo pip install cvxpy で追加インストール可能")

# ============================================================================
# 3. パフォーマンス分析
# ============================================================================
print(f"\n⚡ 3. 大規模問題とパフォーマンス分析")
print("-" * 40)

import time

def benchmark_optimization_methods(scenario_sizes=[10, 50, 100, 200]):
    """異なるシナリオ数での最適化性能を比較"""
    benchmark_results = []
    
    for n_scenarios in scenario_sizes:
        print(f"シナリオ数 {n_scenarios} での性能測定...")
        
        # シナリオ生成
        np.random.seed(42)
        wind_scenarios = np.random.normal(50, 10, n_scenarios)
        probabilities = np.ones(n_scenarios) / n_scenarios
        
        try:
            # CVaR最適化のベンチマーク
            optimizer = CVaROptimizer(wind_scenarios, probabilities)
            
            start_time = time.time()
            cvar_result = optimizer.solve_cvar_optimization()
            cvar_time = time.time() - start_time
            
            start_time = time.time()
            ev_result = optimizer.solve_expected_value_optimization()
            ev_time = time.time() - start_time
            
            if cvar_result['status'] == 'optimal' and ev_result['status'] == 'optimal':
                benchmark_results.append({
                    'scenarios': n_scenarios,
                    'cvar_time': cvar_time,
                    'cvar_objective': cvar_result['cvar_value'],
                    'cvar_x': cvar_result['optimal_x'],
                    'ev_time': ev_time,
                    'ev_objective': ev_result['expected_cost'],
                    'ev_x': ev_result['optimal_x']
                })
                
                print(f"  CVaR最適化: {cvar_time:.4f}秒")
                print(f"  期待値最適化: {ev_time:.4f}秒")
            else:
                print(f"  最適化失敗")
                
        except:
            print(f"  エラー発生")
    
    return pd.DataFrame(benchmark_results)

# ベンチマーク実行
if 'cp' in locals():
    benchmark_df = benchmark_optimization_methods()
    
    if not benchmark_df.empty:
        print(f"\n=== パフォーマンス分析結果 ===")
        print(benchmark_df.round(4))
        
        # 統計分析
        if len(benchmark_df) > 1:
            time_ratio = benchmark_df['cvar_time'].iloc[-1] / benchmark_df['cvar_time'].iloc[0]
            scenario_ratio = benchmark_df['scenarios'].iloc[-1] / benchmark_df['scenarios'].iloc[0]
            
            print(f"\n計算複雑度分析:")
            print(f"シナリオ数が{scenario_ratio:.0f}倍になると計算時間は{time_ratio:.1f}倍")
            
            # 実用的な推奨値
            max_time = benchmark_df['cvar_time'].max()
            if max_time < 1.0:
                print(f"✓ 最大計算時間{max_time:.3f}秒 - リアルタイム運用に適用可能")
            elif max_time < 10.0:
                print(f"⚠️  最大計算時間{max_time:.3f}秒 - 短期計画に適用可能") 
            else:
                print(f"⚠️  最大計算時間{max_time:.3f}秒 - 長期計画のみ適用可能")
else:
    print("CVXPYが利用不可のためパフォーマンス分析をスキップ")

# ============================================================================
# 4. 実用的な最適化戦略とまとめ
# ============================================================================
print(f"\n💡 4. 実用的な最適化戦略")
print("-" * 40)

strategies = {
    "リスク中立": {
        "手法": "期待値最適化",
        "適用場面": "安定した運用環境、リスク許容度が高い",
        "計算負荷": "低",
        "実装難易度": "易"
    },
    "リスク回避（軽度）": {
        "手法": "CVaR最適化（95%信頼水準）",
        "適用場面": "一般的な電力系統運用",
        "計算負荷": "中",
        "実装難易度": "中"
    },
    "リスク回避（強度）": {
        "手法": "CVaR最適化（99%信頼水準）",
        "適用場面": "重要負荷、高信頼性要求",
        "計算負荷": "中",
        "実装難易度": "中"
    },
    "投資計画": {
        "手法": "二段階確率計画法",
        "適用場面": "長期設備投資、不確実性の高い将来計画",
        "計算負荷": "高",
        "実装難易度": "難"
    }
}

strategy_df = pd.DataFrame.from_dict(strategies, orient='index')
print(f"\n=== 戦略比較表 ===")
print(strategy_df)

print(f"\n=== 実装上の考慮事項 ===")
considerations = [
    "1. シナリオ生成: 歴史データ vs Monte Carlo vs 専門家判断",
    "2. 計算時間: リアルタイム運用では数秒、計画問題では数分〜数時間",
    "3. ソルバー選択: オープンソース(OSQP, CBC) vs 商用(Gurobi, CPLEX)",
    "4. スケーラビリティ: シナリオ削減、分解手法の活用",
    "5. 感度分析: パラメータの不確実性への対応",
    "6. 検証: バックテスト、アウトオブサンプル検証"
]

for consideration in considerations:
    print(consideration)

print(f"\n=== 学習成果 ===")
learning_outcomes = [
    "✓ CVaR最適化の実装と活用",
    "✓ シナリオベース確率計画法", 
    "✓ 二段階確率計画法による投資・運用統合最適化",
    "✓ 大規模問題の性能分析とスケーラビリティ",
    "✓ 実用的な最適化戦略の選択指針",
    "✓ CVXPYとPuLPを用いた高度な最適化実装"
]

for outcome in learning_outcomes:
    print(outcome)

print(f"\n" + "="*60)
print("✅ 高度最適化ノートブック実行完了!")
print("="*60)