# CIGRE TB820 確率計画法 練習問題集

CIGRE Technical Brochure 820「Stochastic planning methods for electricity transmission grids」に基づいた確率計画法の練習問題集です。

## 概要

本リポジトリでは、電力系統の確率計画法について体系的に学習できる練習問題を提供しています。理論的な理解から実装まで段階的に習得できるよう構成されています。

## オンライン版リンク

- **基礎編**: https://www.genspark.ai/doc_agent?id=69b78e97-1097-4fed-8e37-6b801a8520bd
- **Python実装編**: https://www.genspark.ai/doc_agent?id=2abd4cd4-e4f1-450a-a470-db2f6952014e

これらのリンクは常に最新の内容を表示します。

## 構成

```
CIGRE_stochastic_TB820/
├── README.md                          # このファイル
├── GITHUB_SETUP.md                    # GitHubアップロード手順
├── 01-basic-exercises/                # 基礎編：理論的な問題
│   ├── README.md
│   └── problems/
│       ├── problem_01.md
│       ├── problem_02.md
│       └── ...
├── 02-python-implementation/          # Python実装編
│   ├── README.md
│   ├── src/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   └── models/
│   ├── notebooks/
│   └── data/
├── data/                              # サンプルデータ
└── references/                        # 参考資料
```

## サンプル問題（クイックスタート）

### 問題1: 基本的な確率計画問題

**設定**:
- 変動する再生可能エネルギー発電量を考慮した発電計画
- 風力発電の出力は正規分布 N(μ=50MW, σ²=100MW²) に従う
- 従来発電機のコストは 60$/MWh

**問題**: 
期待コストを最小化する最適な発電スケジュールを求めよ。

**目標**:
- 確率分布の理解
- 期待値計算
- 最適化問題の定式化

### 問題2: リスク制約付き最適化

**設定**:
- 送電線の熱容量制約: 100MW
- 負荷の不確実性: 一様分布 U(80, 120)MW
- 制約違反確率の上限: 5%

**問題**:
制約違反確率を5%以下に保ちつつ、運用コストを最小化せよ。

**目標**:
- 確率制約の理解
- CVaR (Conditional Value at Risk) の概念
- リスク管理

### 問題3: シナリオベース最適化

**設定**:
- 5つの需要シナリオ（各20%の確率）
- 3つの発電機オプション
- 投資と運用のトレードオフ

**問題**:
期待総コストを最小化する投資・運用計画を求めよ。

**目標**:
- シナリオ分析手法
- 二段階最適化
- 投資決定問題

## 学習パス

### 初級（理論重視）
1. `01-basic-exercises/problems/problem_01-05.md` - 基本概念
2. 確率分布と統計的指標の理解
3. 手計算による小規模問題の解法

### 中級（実装重視）
1. `02-python-implementation/notebooks/` - Python実装
2. SciPy/NumPy による最適化
3. Monte Carlo シミュレーション

### 上級（応用問題）
1. 大規模システムのモデリング
2. 高速解法アルゴリズム
3. 実際の系統データを用いた検証

## 必要な前提知識

- 線形代数の基礎
- 確率・統計の基本概念
- 最適化理論の基礎
- Python プログラミング（実装編）

## 使用するツール・ライブラリ

- **Python**: NumPy, SciPy, pandas, matplotlib
- **最適化ソルバー**: CVXPY, PuLP, Gurobi（オプション）
- **可視化**: matplotlib, seaborn, plotly

## インストールと実行

```bash
# リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/CIGRE_stochastic_TB820.git
cd CIGRE_stochastic_TB820

# Python環境のセットアップ
pip install -r requirements.txt

# Jupyter Notebookの起動
jupyter notebook 02-python-implementation/notebooks/
```

## 貢献とフィードバック

- 新しい問題の提案
- 解法の改善案
- バグ報告
- ドキュメントの改善

GitHubのIssuesまたはPull Requestsをご利用ください。

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 参考文献

1. CIGRE Working Group C1.1.41, "Stochastic planning methods for electricity transmission grids", Technical Brochure 820, 2020.
2. Conejo, A. J., et al., "Decision making under uncertainty in electricity markets", Springer, 2010.
3. Birge, J. R., & Louveaux, F., "Introduction to stochastic programming", Springer Science & Business Media, 2011.

---

**質問やサポートが必要な場合は、GitHubのIssuesでお知らせください。**