"""
CIGRE TB820 確率計画法 - ユーティリティ関数

確率計画法で使用する基本的な関数を提供
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Union


def calculate_expected_cost(wind_output: np.ndarray, probabilities: np.ndarray, 
                          conventional_gen: float, demand: float = 80.0,
                          conv_cost: float = 60.0, emergency_cost: float = 200.0) -> float:
    """
    期待コストを計算する関数
    
    Parameters:
    -----------
    wind_output : np.ndarray
        風力発電出力のシナリオ [MW]
    probabilities : np.ndarray
        各シナリオの発生確率
    conventional_gen : float
        従来発電機の出力 [MW]
    demand : float
        電力需要 [MW] (デフォルト: 80MW)
    conv_cost : float
        従来発電機のコスト [$/MWh] (デフォルト: 60$/MWh)
    emergency_cost : float
        緊急電源のコスト [$/MWh] (デフォルト: 200$/MWh)
    
    Returns:
    --------
    float
        期待総コスト [$]
    """
    conventional_cost = conv_cost * conventional_gen
    
    # 各シナリオでの不足電力量を計算
    shortage = np.maximum(0, demand - conventional_gen - wind_output)
    
    # 各シナリオでの緊急電源コスト
    emergency_costs = emergency_cost * shortage
    
    # 期待緊急電源コスト
    expected_emergency_cost = np.sum(probabilities * emergency_costs)
    
    return conventional_cost + expected_emergency_cost


def plot_wind_distribution(mu: float = 50.0, sigma: float = 10.0) -> None:
    """
    風力発電出力の確率分布をプロット
    
    Parameters:
    -----------
    mu : float
        平均値 [MW]
    sigma : float
        標準偏差 [MW]
    """
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label=f'N({mu}, {sigma}²)')
    plt.axvline(mu, color='r', linestyle='--', alpha=0.7, label=f'平均値: {mu}MW')
    plt.xlabel('風力発電出力 [MW]')
    plt.ylabel('確率密度')
    plt.title('風力発電出力の確率分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_var_cvar(losses: np.ndarray, probabilities: np.ndarray, 
                      confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    VaRとCVaRを計算
    
    Parameters:
    -----------
    losses : np.ndarray
        損失のシナリオ
    probabilities : np.ndarray
        各シナリオの確率
    confidence_level : float
        信頼水準 (デフォルト: 0.95 = 95%)
    
    Returns:
    --------
    Tuple[float, float]
        (VaR, CVaR)
    """
    # ソートしたインデックスを取得
    sorted_indices = np.argsort(losses)
    sorted_losses = losses[sorted_indices]
    sorted_probs = probabilities[sorted_indices]
    
    # 累積確率を計算
    cumulative_probs = np.cumsum(sorted_probs)
    
    # VaRを計算（信頼水準での分位点）
    var_index = np.searchsorted(cumulative_probs, confidence_level)
    if var_index >= len(sorted_losses):
        var = sorted_losses[-1]
    else:
        var = sorted_losses[var_index]
    
    # CVaRを計算（VaRを超える損失の条件付期待値）
    beyond_var_mask = sorted_losses >= var
    if np.any(beyond_var_mask):
        beyond_var_probs = sorted_probs[beyond_var_mask]
        beyond_var_losses = sorted_losses[beyond_var_mask]
        
        # 正規化された確率で加重平均
        total_prob = np.sum(beyond_var_probs)
        if total_prob > 0:
            cvar = np.sum(beyond_var_probs * beyond_var_losses) / total_prob
        else:
            cvar = var
    else:
        cvar = var
    
    return var, cvar


def solve_basic_stochastic_problem() -> dict:
    """
    基本的な確率計画問題を解く（問題1のサンプル）
    
    Returns:
    --------
    dict
        最適解と関連情報
    """
    # 離散化された風力出力シナリオ
    wind_scenarios = np.array([30.0, 50.0, 70.0])  # MW
    probabilities = np.array([0.3, 0.4, 0.3])
    
    # 従来発電機の出力候補
    conv_gen_candidates = np.linspace(0, 80, 81)
    
    # 各候補での期待コストを計算
    expected_costs = [
        calculate_expected_cost(wind_scenarios, probabilities, conv_gen)
        for conv_gen in conv_gen_candidates
    ]
    
    # 最適解を見つける
    optimal_index = np.argmin(expected_costs)
    optimal_conv_gen = conv_gen_candidates[optimal_index]
    optimal_cost = expected_costs[optimal_index]
    
    return {
        'optimal_conventional_generation': optimal_conv_gen,
        'optimal_expected_cost': optimal_cost,
        'wind_scenarios': wind_scenarios,
        'probabilities': probabilities,
        'all_costs': np.array(expected_costs),
        'conv_gen_candidates': conv_gen_candidates
    }


if __name__ == "__main__":
    # サンプル実行
    print("CIGRE TB820 確率計画法 - サンプル計算")
    print("=" * 50)
    
    # 基本問題を解く
    result = solve_basic_stochastic_problem()
    
    print(f"最適な従来発電機出力: {result['optimal_conventional_generation']:.1f} MW")
    print(f"最適期待コスト: ${result['optimal_expected_cost']:.2f}")
    print()
    
    # 風力分布をプロット
    plot_wind_distribution()
    
    # VaR/CVaR計算例
    losses = np.array([100, 200, 300, 400, 500])
    probs = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
    var, cvar = calculate_var_cvar(losses, probs, 0.95)
    
    print(f"VaR (95%): ${var:.2f}")
    print(f"CVaR (95%): ${cvar:.2f}")