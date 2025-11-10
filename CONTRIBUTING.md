# 貢献ガイド

CIGRE TB820確率計画法練習問題集への貢献をありがとうございます！

## 貢献方法

### 1. 新しい問題の追加

- `01-basic-exercises/problems/` に新しいmarkdownファイルを作成
- 問題設定、解答の方向性、学習ポイントを含める
- 理論的説明と実践的応用のバランスを保つ

### 2. Python実装の改善

- `02-python-implementation/src/` のコード最適化
- 新しいアルゴリズムの実装
- バグ修正とパフォーマンス改善

### 3. ドキュメントの改善

- README.mdの更新
- コメントの追加と改善
- 使用例の充実

## 開発環境のセットアップ

```bash
# リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/CIGRE_stochastic_TB820.git
cd CIGRE_stochastic_TB820

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

## コーディング規則

### Python

- PEP 8に従う
- 関数とクラスにdocstringを追加
- 型ヒントを使用する
- テストを書く（可能な場合）

### Markdown

- 見出しの階層を適切に使用
- コードブロックに言語を指定
- 数式はLaTeX記法を使用

## プルリクエストのガイドライン

1. **明確な説明**: 変更内容と理由を詳しく説明
2. **小さな変更**: 一つのPRで一つの機能/修正に集中
3. **テスト**: 可能な限りテストを含める
4. **ドキュメント**: 新機能にはドキュメントを追加

## 問題報告

Issuesで以下を報告してください：

- バグや間違い
- 改善提案
- 新機能のアイデア
- ドキュメントの不備

## ライセンス

貢献されたコードはMITライセンスの下でリリースされます。

## 質問

質問がある場合は、Issuesで気軽にお尋ねください。