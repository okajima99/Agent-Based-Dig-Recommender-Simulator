Agent-Based-Dig-Recommender-Simulator
本コードは、Infinite Scroll 型プラットフォーム上でのユーザーのディグ行動を再現するエージェントベース・シミュレータです。各エージェントは G（ジャンル嗜好ベクトル）・V（価値観ベクトル）・I（本能ベクトル）を持ち、ユーザーとコンテンツで相互作用します。コンテンツ表示アルゴリズムは環境変数 DISPLAY_ALGORITHM で切り替え可能で、“random” / “popularity” / “trend” / “buzz” / “cbf_item” / “cbf_user” / “cf_item” / “cf_user” を実装しています。“cbf_item or user” / “cf_item or user”にはaffinityとnoveltyフラグを用意しています。各ステップでエージェントは提示コンテンツに対して視聴・いいね・掘りを確率的に選択し、その結果に応じて G/V が微調整されます。PyTorchによりランキング計算を高速化し、探索用の追加ランダムブロックやコンテンツ補充の設定もパラメータで制御できます。
⸻

■ 必要な環境

・Python 3.10 以上
・PyTorch 2.x（CUDA または MPS 対応版推奨）
・NumPy / Matplotlib

推奨環境：
・NVIDIA GPU (CUDA 11+) → ランキング処理が高速化

⸻

■ セットアップ
	1.	リポジトリをクローン
git clone 
	2.	ディレクトリへ移動
cd 
	3.	必要なライブラリをインストール
pip install -r requirements.txt
	4.	実行
python Agent-Based-Dig-Recommender-Simulator.py

requirements.txt の例：
torch
numpy
matplotlib

⸻

■ プロジェクト構成（概要）

Agent-Based-Dig-Recommender-Simulator.py　…　メインシミュレータ
data/　…　ログや統計出力（任意）
README.md

本ファイルのみで、以下の処理が完結します：
・エージェント（G/V）の状態管理と更新
・推薦アルゴリズム（random / popularity / trend / buzz / CBF item or user(affinity or novelty) / CF item or user(affinity or novelty) ）
・GPU高速化された Top-K ランキング
・視聴／いいね／Dig の確率判定
・ログ生成・統計集計

⸻

■ 出力されるデータ

・各ステップの視聴数 / いいね数 / Dig数
・アルゴリズム別 Dig率・Dig時の平均順位
・ユーザーG–コンテンツG の Pearson相関の推移
・エージェントの G/Vベクトルの変化
・推薦方式別の探索傾向の比較

研究・可視化・アルゴリズム比較に直接利用できます。

■ ライセンス
必要に応じて MIT / Apache-2.0 / CC-BY などを設定してください。
