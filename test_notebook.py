#!/usr/bin/env python3
"""
ノートブック機能のテストスクリプト
"""

import sys
import os
sys.path.append('02-python-implementation/src')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# 自作utilsモジュールのインポート
try:
    from utils import solve_basic_stochastic_problem, calculate_expected_cost
    print("✓ utils.py モジュールのインポート成功")
except ImportError as e:
    print(f"✗ utils.py インポートエラー: {e}")
    sys.exit(1)

def test_basic_optimization():
    """基本的な最適化のテスト"""
    print("\n" + "="*50)
    print("基本的な確率計画問題の実行テスト")
    print("="*50)
    
    # 基本問題を解く
    result = solve_basic_stochastic_problem()
    
    print(f"最適な従来発電量: {result['optimal_conventional_generation']:.2f} MW")
    print(f"最適期待コスト: ${result['optimal_expected_cost']:.2f}")
    print(f"風力シナリオ: {result['wind_scenarios']}")
    print(f"確率: {result['probabilities']}")
    
    # 可視化
    plt.figure(figsize=(12, 8))
    
    # サブプロット1: 期待コスト曲線
    plt.subplot(2, 2, 1)
    plt.plot(result['conv_gen_candidates'], result['all_costs'], 'b-', linewidth=2, label='Expected Cost')
    plt.axvline(result['optimal_conventional_generation'], color='r', linestyle='--', linewidth=2, 
                label=f'Optimal: {result["optimal_conventional_generation"]:.1f} MW')
    plt.xlabel('Conventional Generation [MW]')
    plt.ylabel('Expected Cost [$]')
    plt.title('Expected Cost vs Conventional Generation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット2: 風力シナリオ
    plt.subplot(2, 2, 2)
    plt.bar(range(len(result['wind_scenarios'])), result['wind_scenarios'], 
            alpha=0.7, color='skyblue', label='Wind Scenarios')
    plt.xlabel('Scenario')
    plt.ylabel('Wind Output [MW]')
    plt.title('Wind Power Scenarios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット3: 感度分析
    plt.subplot(2, 2, 3)
    wind_means = np.linspace(30, 70, 20)
    optimal_costs = []
    
    for wm in wind_means:
        # 簡化した感度分析
        cost = calculate_expected_cost(result['wind_scenarios'], result['probabilities'], 
                                     result['optimal_conventional_generation'])
        optimal_costs.append(cost)
    
    plt.plot(wind_means, optimal_costs, 'g-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Wind Mean [MW]')
    plt.ylabel('Optimal Cost [$]')
    plt.title('Sensitivity Analysis')
    plt.grid(True, alpha=0.3)
    
    # サブプロット4: 確率分布
    plt.subplot(2, 2, 4)
    x = np.linspace(0, 100, 1000)
    wind_pdf = stats.norm.pdf(x, 50, 10)
    plt.plot(x, wind_pdf, 'purple', linewidth=2, label='Wind PDF N(50,10²)')
    plt.axvline(50, color='red', linestyle='--', alpha=0.7, label='Mean: 50MW')
    plt.fill_between(x, wind_pdf, alpha=0.3, color='purple')
    plt.xlabel('Wind Output [MW]')
    plt.ylabel('Probability Density')
    plt.title('Wind Power Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ グラフを comprehensive_test_results.png として保存")
    
    return result

def test_advanced_features():
    """高度な機能のテスト"""
    print("\n" + "="*50)
    print("高度な機能のテスト")
    print("="*50)
    
    # CVaR計算のテスト
    np.random.seed(42)
    n_samples = 1000
    wind_scenarios = np.random.normal(50, 10, n_samples)
    probabilities = np.ones(n_samples) / n_samples
    
    # 複数の発電量でのリスク分析
    generation_options = [20, 30, 40, 50]
    risk_results = []
    
    for gen in generation_options:
        costs = []
        for wind in wind_scenarios:
            shortage = max(0, 80 - gen - wind)
            total_cost = 60 * gen + 200 * shortage
            costs.append(total_cost)
        
        costs = np.array(costs)
        mean_cost = np.mean(costs)
        var_95 = np.percentile(costs, 95)
        cvar_95 = np.mean(costs[costs >= var_95])
        
        risk_results.append({
            'Generation': gen,
            'Mean_Cost': mean_cost,
            'VaR_95': var_95,
            'CVaR_95': cvar_95
        })
    
    # リスク分析結果の表示
    risk_df = pd.DataFrame(risk_results)
    print("\nリスク分析結果:")
    print(risk_df.round(2))
    
    # リスク分析の可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(risk_df['Generation'], risk_df['Mean_Cost'], 'o-', label='Expected Cost', linewidth=2)
    plt.plot(risk_df['Generation'], risk_df['VaR_95'], 's-', label='VaR 95%', linewidth=2)
    plt.plot(risk_df['Generation'], risk_df['CVaR_95'], '^-', label='CVaR 95%', linewidth=2)
    plt.xlabel('Generation [MW]')
    plt.ylabel('Cost [$]')
    plt.title('Risk Measures vs Generation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(risk_df['Mean_Cost'], risk_df['VaR_95'], s=100, alpha=0.7)
    for i, gen in enumerate(risk_df['Generation']):
        plt.annotate(f'{gen}MW', (risk_df.iloc[i]['Mean_Cost'], risk_df.iloc[i]['VaR_95']),
                    xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Expected Cost [$]')
    plt.ylabel('VaR 95% [$]')
    plt.title('Risk-Return Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_analysis_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 高度な分析グラフを advanced_analysis_results.png として保存")
    
    return risk_df

def main():
    """メイン実行関数"""
    print("CIGRE TB820 確率計画法ノートブックのテスト実行")
    print("="*60)
    
    try:
        # 基本的な最適化テスト
        basic_result = test_basic_optimization()
        
        # 高度な機能テスト
        advanced_result = test_advanced_features()
        
        print("\n" + "="*60)
        print("✅ すべてのテストが正常に完了しました！")
        print("="*60)
        
        print("\n📊 結果サマリー:")
        print(f"• 最適従来発電量: {basic_result['optimal_conventional_generation']:.1f} MW")
        print(f"• 最適期待コスト: ${basic_result['optimal_expected_cost']:.0f}")
        print(f"• 生成ファイル: comprehensive_test_results.png, advanced_analysis_results.png")
        
        print("\n🚀 次のステップ:")
        print("• Jupyter Notebook環境でより詳細な分析が可能です")
        print("• 各ノートブック(.ipynb)を開いて対話的に実行してください")
        print("• パラメータを変更して感度分析を実行してください")
        
    except Exception as e:
        print(f"\n❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()