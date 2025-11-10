# GitHubへのアップロード手順

このリポジトリをGitHubにアップロードする手順を説明します。

## 前提条件

- GitHubアカウントを持っていること
- Git がインストールされていること（このリポジトリは既に初期化済み）

## 手順

### 1. GitHubで新しいリポジトリを作成

1. GitHubにログイン
2. 右上の「+」→「New repository」をクリック
3. リポジトリ情報を入力：
   - **Repository name**: `stochastic-programming-exercises`（任意の名前でOK）
   - **Description**: `CIGRE TB820に基づいた確率計画法の練習問題集`
   - **Visibility**: Public または Private を選択
   - **Initialize this repository with**: 何もチェックしない（すでにローカルに内容があるため）
4. 「Create repository」をクリック

### 2. リモートリポジトリを追加

GitHubで作成したリポジトリのURLをコピーして、以下のコマンドを実行：

```bash
cd /home/user/stochastic-programming-exercises

# GitHubのリポジトリURLに置き換えてください
git remote add origin https://github.com/YOUR_USERNAME/stochastic-programming-exercises.git

# または SSH の場合
# git remote add origin git@github.com:YOUR_USERNAME/stochastic-programming-exercises.git
```

### 3. リポジトリをプッシュ

```bash
# mainブランチをプッシュ
git push -u origin main
```

### 4. 認証

初回プッシュ時には認証が必要です：

#### HTTPSの場合
- ユーザー名とパスワード（Personal Access Token）を入力
- Personal Access Tokenの作成方法：
  1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  2. 「Generate new token」→「Generate new token (classic)」
  3. 権限で「repo」にチェック
  4. 生成されたトークンをコピー（一度しか表示されません）
  5. パスワードの代わりにこのトークンを使用

#### SSHの場合
- SSH鍵を事前に登録しておく必要があります
- 詳細：https://docs.github.com/ja/authentication/connecting-to-github-with-ssh

### 5. 確認

GitHubのリポジトリページにアクセスして、ファイルがアップロードされていることを確認します。

## プッシュ後の構成

```
GitHub Repository
├── README.md                          # プロジェクト全体の説明
├── LICENSE                            # MITライセンス
├── .gitignore                         # 除外ファイル設定
├── CONTRIBUTING.md                    # 貢献ガイド
├── GITHUB_SETUP.md                    # このファイル
├── 01-basic-exercises/                # 基礎編
│   ├── README.md
│   └── DOWNLOAD_LINK.md
├── 02-python-implementation/          # Python実装編
│   ├── README.md
│   ├── DOWNLOAD_LINK.md
│   └── src/
│       ├── __init__.py
│       └── utils.py
├── data/                              # サンプルデータ
├── references/                        # 参考資料
└── notebooks/                         # Jupyter Notebook（後で追加）
```

## 今後の更新

新しいファイルを追加したり、既存のファイルを更新した場合：

```bash
cd /home/user/stochastic-programming-exercises

# 変更を確認
git status

# ファイルを追加
git add .

# コミット
git commit -m "変更内容の説明"

# プッシュ
git push origin main
```

## よくある問題と解決方法

### 問題1: 認証エラー

```
error: failed to push some refs to 'https://github.com/...'
```

**解決方法**: Personal Access Tokenを使用しているか確認。パスワードは使用できません。

### 問題2: リモートとの競合

```
error: failed to push some refs to '...'
hint: Updates were rejected because the remote contains work that you do not have locally.
```

**解決方法**:
```bash
git pull origin main --rebase
git push origin main
```

### 問題3: ファイルサイズが大きすぎる

GitHubは100MBを超えるファイルをプッシュできません。

**解決方法**: 大きなファイルは.gitignoreに追加し、別の方法で共有する。

## オンラインドキュメントのリンク

README.mdにオンライン版のリンクが含まれています：

- **基礎編**: https://www.genspark.ai/doc_agent?id=69b78e97-1097-4fed-8e37-6b801a8520bd
- **Python実装編**: https://www.genspark.ai/doc_agent?id=2abd4cd4-e4f1-450a-a470-db2f6952014e

これらのリンクは常に最新の内容を表示します。

## 次のステップ

1. GitHubリポジトリに README を充実させる
2. 問題のコードをJupyter Notebookに変換
3. サンプルデータを追加
4. GitHub Pagesでドキュメントを公開（オプション）
5. GitHub Actionsでテストを自動化（オプション）

---

**質問やサポートが必要な場合は、GitHubのIssuesでお知らせください。**
