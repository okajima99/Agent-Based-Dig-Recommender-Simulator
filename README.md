# Agent-Based-Dig-Recommender-Simulator
本コードは、Infinite Scroll型サービスでのユーザーの“ディグ行動”を再現するエージェントベース・シミュレーション。G・Vベクトルで内的嗜好を表し、推薦アルゴリズム（random／popularity／trend／CBF／CF／novelty など）に基づきコンテンツを提示。エージェントは視聴・Like・Dig を確率的に判断し、状態更新をGPU（MPS/CUDA）で高速実行。ログを集計し、RecBole学習用データも生成する総合的な研究用フレームワーク。
