#!/usr/bin/env python3
"""
基礎編ノートブック (01_basic_stochastic_optimization.ipynb) の主要部分を実行
"""

import sys
import os
sys.path.append('02-python-implementation/src')

# 基本ライブラリ
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, optimize
import seaborn as sns

# 設定
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.grid'] = True

print("🚀 CIGRE TB820 基礎編ノートブック実行開始")
print("="*60)

# ============================================================================
# 1. 問題パラメータの定義
# ============================================================================
print("\n📊 1. 問題設定")
print("-" * 40)

# 風力発電の分布パラメータ
wind_mean = 50.0  # MW
wind_std = 10.0   # MW

# コストパラメータ
conv_cost = 60.0      # $/MWh
emergency_cost = 200.0 # $/MWh

# 電力需要
demand = 80.0  # MW

print(f"風力発電: N({wind_mean}, {wind_std}²) MW")
print(f"従来発電コスト: {conv_cost} $/MWh")
print(f"緊急電源コスト: {emergency_cost} $/MWh")
print(f"電力需要: {demand} MW")

# ============================================================================
# 2. 風力発電出力の確率分布分析
# ============================================================================
print("\n📈 2. 風力発電出力の確率分布分析")
print("-" * 40)

# 風力発電出力の範囲を設定
wind_range = np.linspace(wind_mean - 4*wind_std, wind_mean + 4*wind_std, 1000)

# 確率密度関数
wind_pdf = stats.norm.pdf(wind_range, wind_mean, wind_std)

# 累積分布関数
wind_cdf = stats.norm.cdf(wind_range, wind_mean, wind_std)

# 統計量の計算
print(f"平均: {wind_mean} MW")
print(f"標準偏差: {wind_std} MW")
print(f"95%信頼区間: [{wind_mean - 1.96*wind_std:.1f}, {wind_mean + 1.96*wind_std:.1f}] MW")

# ============================================================================
# 3. 期待コスト関数の定義と計算
# ============================================================================
print("\n🧮 3. 期待コスト関数の実装")
print("-" * 40)

def expected_cost_analytical(x, wind_mean=50, wind_std=10, 
                           conv_cost=60, emergency_cost=200, demand=80):
    """
    解析的な期待コスト計算（正規分布の場合）
    """
    # 従来発電機のコスト
    conventional_cost = conv_cost * x
    
    # 不足分の閾値
    threshold = demand - x
    
    # 不足確率: P(W < threshold)
    shortage_prob = stats.norm.cdf(threshold, wind_mean, wind_std)
    
    # 条件付き期待不足量: E[threshold - W | W < threshold]
    if shortage_prob > 0:
        # 切断正規分布の期待値
        standardized_threshold = (threshold - wind_mean) / wind_std
        phi = stats.norm.pdf(standardized_threshold)
        Phi = stats.norm.cdf(standardized_threshold)
        
        if Phi > 1e-10:  # 数値安定性のため
            expected_shortage = (threshold - wind_mean) + wind_std * phi / Phi
        else:
            expected_shortage = 0
    else:
        expected_shortage = 0
    
    # 期待緊急電源コスト
    expected_emergency_cost = emergency_cost * shortage_prob * expected_shortage
    
    return conventional_cost + expected_emergency_cost

# テスト
x_test = 30.0
cost_test = expected_cost_analytical(x_test)
print(f"従来発電 {x_test} MW の期待コスト: ${cost_test:.2f}")

# ============================================================================
# 4. 最適化実行
# ============================================================================
print("\n🎯 4. 期待コスト最適化")
print("-" * 40)

# 従来発電機の出力範囲
x_range = np.linspace(0, 80, 200)

# 各出力での期待コストを計算
costs_analytical = [expected_cost_analytical(x) for x in x_range]

# 最適化（解析解）
result = optimize.minimize_scalar(expected_cost_analytical, bounds=(0, 100), method='bounded')
optimal_x = result.x
optimal_cost = result.fun

print(f"最適な従来発電量: {optimal_x:.2f} MW")
print(f"最適期待コスト: ${optimal_cost:.2f}")
print(f"従来発電コスト: ${conv_cost * optimal_x:.2f}")
print(f"期待緊急電源コスト: ${optimal_cost - conv_cost * optimal_x:.2f}")

# ============================================================================
# 5. 可視化
# ============================================================================
print("\n📊 5. 結果の可視化")
print("-" * 40)

# 2x2のサブプロット
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. 風力分布 (PDF)
ax1.plot(wind_range, wind_pdf, 'b-', linewidth=2, label=f'N({wind_mean}, {wind_std}²)')
ax1.axvline(wind_mean, color='r', linestyle='--', alpha=0.7, label=f'平均: {wind_mean}MW')
ax1.axvline(wind_mean - wind_std, color='orange', linestyle=':', alpha=0.7, label='±1σ')
ax1.axvline(wind_mean + wind_std, color='orange', linestyle=':', alpha=0.7)
ax1.fill_between(wind_range, wind_pdf, alpha=0.3)
ax1.set_xlabel('Wind Output [MW]')
ax1.set_ylabel('Probability Density')
ax1.set_title('Wind Output Distribution (PDF)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 期待コストカーブ
ax2.plot(x_range, costs_analytical, 'b-', linewidth=2, label='Expected Total Cost')
ax2.axvline(optimal_x, color='r', linestyle='--', linewidth=2, 
            label=f'Optimal x = {optimal_x:.1f} MW')
ax2.scatter([optimal_x], [optimal_cost], color='red', s=100, zorder=5)

# コスト成分の分析
conv_costs = conv_cost * x_range
emergency_costs = [expected_cost_analytical(x) - conv_cost*x for x in x_range]

ax2.plot(x_range, conv_costs, '--', color='green', alpha=0.7, label='Conventional Cost')
ax2.plot(x_range, emergency_costs, '--', color='orange', alpha=0.7, label='Expected Emergency Cost')

ax2.set_xlabel('Conventional Generation [MW]')
ax2.set_ylabel('Cost [$]')
ax2.set_title('Expected Cost vs Conventional Generation')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 感度分析（風力平均値の影響）
wind_means = np.linspace(30, 70, 20)
optimal_x_wind_mean = []

for wm in wind_means:
    result_sens = optimize.minimize_scalar(
        lambda x: expected_cost_analytical(x, wind_mean=wm),
        bounds=(0, 100), method='bounded'
    )
    optimal_x_wind_mean.append(result_sens.x)

ax3.plot(wind_means, optimal_x_wind_mean, 'b-', linewidth=2, marker='o', markersize=4)
ax3.set_xlabel('Wind Mean [MW]')
ax3.set_ylabel('Optimal Conventional Gen [MW]')
ax3.set_title('Sensitivity: Wind Mean vs Optimal Generation')
ax3.grid(True, alpha=0.3)

# 4. リスク分析（VaR/CVaR計算例）
# Monte Carlo シミュレーション
np.random.seed(42)
n_samples = 10000
wind_samples = np.random.normal(wind_mean, wind_std, n_samples)

# 最適解での総コスト分布
shortage_samples = np.maximum(0, demand - optimal_x - wind_samples)
total_costs = conv_cost * optimal_x + emergency_cost * shortage_samples

# VaR/CVaR計算
sorted_costs = np.sort(total_costs)
var_95 = np.percentile(total_costs, 95)
cvar_95 = np.mean(sorted_costs[sorted_costs >= var_95])

ax4.hist(total_costs, bins=50, alpha=0.7, density=True, color='skyblue', label='Cost Distribution')
ax4.axvline(np.mean(total_costs), color='green', linestyle='-', linewidth=2, label=f'Expected: ${np.mean(total_costs):.0f}')
ax4.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'VaR (95%): ${var_95:.0f}')
ax4.axvline(cvar_95, color='darkred', linestyle=':', linewidth=2, label=f'CVaR (95%): ${cvar_95:.0f}')

ax4.set_xlabel('Total Cost [$]')
ax4.set_ylabel('Probability Density')
ax4.set_title('Cost Distribution and Risk Measures')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('basic_notebook_execution_results.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ 可視化完了: basic_notebook_execution_results.png")

# ============================================================================
# 6. 結果サマリーと実践的考察
# ============================================================================
print("\n💡 6. 結果サマリーと考察")
print("-" * 40)

print(f"\n📊 最適化結果:")
print(f"• 最適従来発電量: {optimal_x:.1f} MW")
print(f"• 最適期待コスト: ${optimal_cost:,.0f}")
print(f"• 従来発電コスト: ${conv_cost * optimal_x:,.0f} ({(conv_cost * optimal_x / optimal_cost * 100):.1f}%)")
print(f"• 期待緊急電源コスト: ${optimal_cost - conv_cost * optimal_x:,.0f} ({((optimal_cost - conv_cost * optimal_x) / optimal_cost * 100):.1f}%)")

print(f"\n⚠️  リスク指標:")
print(f"• 期待コスト: ${np.mean(total_costs):,.0f}")
print(f"• VaR (95%): ${var_95:,.0f}")
print(f"• CVaR (95%): ${cvar_95:,.0f}")
print(f"• リスクプレミアム: ${cvar_95 - np.mean(total_costs):,.0f}")

print(f"\n🔍 感度分析:")
slope = (optimal_x_wind_mean[-1] - optimal_x_wind_mean[0]) / (wind_means[-1] - wind_means[0])
print(f"• 風力平均が1MW増加 → 最適従来発電が{-slope:.2f}MW減少")

# 不足確率の計算
shortage_prob = stats.norm.cdf(demand - optimal_x, wind_mean, wind_std)
print(f"• 電力不足発生確率: {shortage_prob:.1%}")

print(f"\n💭 実践的示唆:")
insights = [
    f"1. 風力平均{wind_mean}MWに対し、最適従来発電は{optimal_x:.1f}MW（需要の{optimal_x/demand:.1%}）",
    f"2. 緊急電源依存度は{((optimal_cost - conv_cost * optimal_x) / optimal_cost):.1%}と適切な水準",
    f"3. 不足確率{shortage_prob:.1%}は許容範囲内",
    f"4. VaR-期待値比は{var_95/np.mean(total_costs):.2f}倍で、リスク管理が重要",
    f"5. 風力予測精度向上により、さらなるコスト削減が期待される"
]

for insight in insights:
    print(f"   {insight}")

print(f"\n🚀 次のステップ:")
next_steps = [
    "• 高度最適化ノートブック(02_advanced_optimization.ipynb)でCVaR最適化を試す",
    "• 可視化ノートブック(03_visualization_analysis.ipynb)でインタラクティブ分析",
    "• パラメータを変更して独自の感度分析を実行",
    "• 実際の風力データでの検証"
]

for step in next_steps:
    print(f"   {step}")

print(f"\n" + "="*60)
print("✅ 基礎編ノートブック実行完了!")
print("="*60)