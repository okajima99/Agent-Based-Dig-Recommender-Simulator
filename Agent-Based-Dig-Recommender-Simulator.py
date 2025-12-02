# ============================================================================
# Imports & 基本セットアップ
# ============================================================================
import random
import os
import time
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
# ----------------------------
# ランダムシードの固定（再現性の確保）
# ----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
# パラメータ設定
# ============================================================================

# --- 実験規模＆ランタイム ---
NUM_GENRES         = 10                      # G（ジャンル）次元
NUM_INSTINCT_DIM   = 5                       # I（本能）次元
NUM_AGENTS         = 1000
NUM_CONTENTS       = 30000
MAX_STEPS          = 100000
INITIAL_RANDOM_STEPS = 0                 # 序盤の完全ランダム表示
# 追加のランダム挿入（探索ウィンドウ）。初期ランダム終了後に適用。
# 例: RANDOM_RANDOM_BLOCK_LEN=300, RANDOM_NORMAL_BLOCK_LEN=1200 → 300ステップ連続ランダム → 1200ステップ通常を繰り返す
RANDOM_INTERVAL_ON       = bool(int(os.getenv("RANDOM_INTERVAL_ON", "1")))
RANDOM_RANDOM_BLOCK_LEN  = int(os.getenv("RANDOM_RANDOM_BLOCK_LEN", "3000"))   # ランダム表示の連続ステップ数
RANDOM_NORMAL_BLOCK_LEN  = int(os.getenv("RANDOM_NORMAL_BLOCK_LEN", "3000"))   # 通常アルゴリズムの連続ステップ数

# --- コンテンツ補充（周期のみ、0指定で無効） ---
# 例: REPLENISH_EVERY=500, REPLENISH_COUNT=10000, START=3000, END=8000
REPLENISH_EVERY        = int(os.getenv("REPLENISH_EVERY", "6000"))  # 0=無効
REPLENISH_COUNT        = int(os.getenv("REPLENISH_COUNT", "60000"))
REPLENISH_START_STEP   = int(os.getenv("REPLENISH_START_STEP", "0"))
REPLENISH_END_STEP     = int(os.getenv("REPLENISH_END_STEP", str(MAX_STEPS)))

# --- 類似度モード ---
#   いいね判定（ユーザー側）は従来通り AGENT_ALPHA を使用可
AGENT_ALPHA  = float(os.getenv("AGENT_ALPHA", "0"))  # いいね判定側のα（維持）
## アルゴリズム側の α は廃止（常にコサイン類似度）

# --- 表示アルゴリズム（環境変数で上書き可能） ---
#   "random" / "popularity" / "trend" / "buzz" / "cbf_item" / "cbf_user" / "cf_item" / "cf_user"
DISPLAY_ALGORITHM = os.getenv("DISPLAY_ALGORITHM", "cbf_item")
# --- アルゴリズムFace（affinity / novelty） ---
#   CBF_ITEM_FACE を正式採用（未指定時は CBF_FACE を後方互換で引き継ぎ）
CBF_ITEM_FACE  = os.getenv("CBF_ITEM_FACE", "affinity")  # "affinity" | "novelty"
CBF_USER_FACE  = os.getenv("CBF_USER_FACE", "affinity")
CF_ITEM_FACE   = os.getenv("CF_ITEM_FACE", "affinity")
CF_USER_FACE   = os.getenv("CF_USER_FACE", "affinity")
# --- 出力ファイルの接頭辞 ---
OUT_PREFIX = os.getenv("OUT_PREFIX", f"simulation_{DISPLAY_ALGORITHM}_noDSSM")
LOG_FILE_PATH = f"{OUT_PREFIX}_analysis.txt"

# --- コンテンツのactive数（G/I） ---
content_max_active   = 3   # G 用
content_i_max_active = 3   # I 用

# --- 生成ポリシー（4値に統一） ---
# 全て: "element" | "norm" | "legacy" | "random"
CONTENT_G_MODE = os.getenv("CONTENT_G_MODE", "random")
CONTENT_I_MODE = os.getenv("CONTENT_I_MODE", "random")
# コンテンツ行列の常駐dtype（cuda時のみ有効）: "fp16" | "bf16"
CONTENT_MAT_DTYPE = os.getenv("CONTENT_MAT_DTYPE", "fp16").lower()

# 既存のエージェント側にも random を追加（既存の3値に+1）
AGENT_G_MODE = os.getenv("AGENT_G_MODE", "element")  # "legacy" | "element" | "norm" | "random"
AGENT_I_MODE = os.getenv("AGENT_I_MODE", "random")  # "legacy" | "element" | "norm" | "random"
AGENT_V_MODE = os.getenv("AGENT_V_MODE", "random")  # "legacy" | "element" | "norm" | "random"

# --- ベクトル生成の分布パラメータ ---
content_G_PARAMS = dict(mu=0.35, sigma=0.30, norm_mu=0.70, norm_sigma=0.27)
content_I_PARAMS = dict(mu=0.35, sigma=0.25, norm_mu=1.30, norm_sigma=0.35)
Agent_G_PARAMS   = dict(mu=0.15, sigma=0.07, norm_mu=0.11, norm_sigma=0.01)
Agent_V_PARAMS   = dict(mu=0.00, sigma=0.50, norm_mu=1.00, norm_sigma=0.25)
Agent_I_PARAMS   = dict(mu=0.35, sigma=0.25, norm_mu=1.30, norm_sigma=0.35)

# --- Like / Dig 関連（ユーザー側は常にdotスケールでロジット） ---
# いいね判定
LOGIT_K   = 13.94
LOGIT_X0  = 0.6108
LIKE_DIVISOR = 3

# 掘り判定
DIG_LOGIT_K  = 16.86
DIG_LOGIT_X0 = 0.2888
DIG_DIVISOR  = 60

# 掘りの更新量（G/V）
DIG_G_STEP  = 0.00115      # 掘りでGを増分
DIG_V_RANGE = 0.10       # 掘りでVを±この範囲でランダム変動

# Like の線形結合重み（CG/CV/I）
LIKE_W_CG = 1.000
LIKE_W_CV = 0.880
LIKE_W_CI = 0.625

# --- 掘りのGauss(d)パラメータ（μ, σ, A に別々の重みを適用） ---
MU0        = 0.1
MU_SLOPE   = 0.2
MU_ALPHA_C = 0.25
MU_BETA_V  = 0.75

SIGMA0        = 0.05
SIGMA_LAMDA   = 2.0
SIGMA_ALPHA_C = 0.25
SIGMA_BETA_V  = 0.75

A0        = 0.75
A_LAMDA   = 0.5

A_ALPHA_C = 0.25
A_BETA_V  = 0.75

# --- グローバル指標（popularity/trend/buzz） ---
# trend
TREND_EMA_ALPHA   = 0.0002
TREND_CACHE_DURATION = 3000

# buzz
BUZZ_WINDOW       = 3000
BUZZ_GAMMA        = 0.9990
BUZZ_CACHE_DURATION = 3000

# popularity
POP_CACHE_DURATION = 3000

# --- 擬似ベクトル/履歴・減衰（要件に沿って整理） ---
PSEUDO_CACHE_DURATION = 720
PSEUDO_DISCOUNT_GAMMA = 0.9998
PSEUDO_HISTORY_WINDOW_STEPS = 10000

# CBF-item 候補上限（0/負で無効＝全件）
CBF_ITEM_TOP_K = 100

# --- CBFユーザー向け内部パラメータ（外部設定なしで固定） ---
CBF_USER_TOP_K_USERS           = 10
CBF_USER_CONTENT_TOP_K         = 10
CBF_USER_TARGET_MIN_CANDIDATES = 1000
CBF_USER_BACKFILL_MAX_RETRIES  = 10
CBF_USER_BACKFILL_GROWTH       = 2
CBF_USER_HISTORY_WINDOW_STEPS  = 10000
CBF_USER_DISCOUNT_GAMMA        = 0.9998

# G/I 融合重み（CBF）
CBF_W_G, CBF_W_I         = 1.0, 1.0
CBF_USER_W_G,  CBF_USER_W_I          = 1.0, 1.0

# CFキャッシュTTL（cf_user / cf_item 兼用）
CF_CACHE_DURATION = 1000
CF_DISCOUNT_GAMMA = 0.9998
CF_HISTORY_WINDOW_STEPS = 10000
CF_NEIGHBOR_TOP_K = 10   # 0/負で無効（全近傍）
CF_CANDIDATE_TOP_K = 10   # 0/負で無効（全候補）

# --- Softmax 温度（各アルゴリズムの確率化強度） ---
LAMBDA_POPULARITY = 20
LAMBDA_TREND      = 20
LAMBDA_BUZZ       = 20
LAMBDA_CBF_ITEM   = 30
LAMBDA_CBF_USER   = 30
LAMBDA_CF_USER    = 30
LAMBDA_CF_ITEM    = 30

# --- ランダム巡回ポリシ ---
RANDOM_REPEAT_POLICY = "reset_when_exhausted"


# --- 5,000刻みスナップショット / パネル設定 ---
STEP_BIN = 6000
PANEL_N_AGENTS = 10
PANEL_AGENT_IDS = list(range(min(PANEL_N_AGENTS, NUM_AGENTS)))

# スナップショット（G多様性の推移）格納先
DIVERSITY_TIMELINE = []  # list of (step, avg_entropy, avg_variance)

# ★追加：G–V 相関のタイムライン（STEP_BINスナップショット）
GV_CORR_TIMELINE = []    # list of (step, avg_pearson_gv, valid_count)
# パネル対象は先頭10人（任意で固定）


# ============================================================================
# ログ出力（コンソールの代わりにテキストへ集約）
# ============================================================================
LOG_LINES: list[str] = []
# LOG_FILE_PATH はパラメータ設定ブロックで定義済み

def log_line(msg: str):
    LOG_LINES.append(str(msg))

def log_and_print(msg: str, *, flush: bool = False):
    """ログに残しつつ即座に標準出力にも流す。"""
    log_line(msg)
    print(msg, flush=flush)

def write_log_file(path: str | None = None):
    path = path or LOG_FILE_PATH
    if path is None:
        return None
    with open(path, "w", encoding="utf-8") as f:
        for line in LOG_LINES:
            f.write(f"{line}\n")
    return path

# ============================================================================
# PyTorch / GPU 設定
# ============================================================================
import torch
# CUDA 優先、無ければCPU
# 乱数の再現性（Torch側）
TORCH_DTYPE = torch.float32
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# CUDA向けの高速設定（TF32等）
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # "medium"でも可
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
# （必要になったら）
# TORCH_GEN = torch.Generator(device=DEVICE)
# TORCH_GEN.manual_seed(RANDOM_SEED)
log_and_print(f"[Torch] version={torch.__version__}, device={DEVICE}, cuda_available={torch.cuda.is_available()}")

def _to_t(x, *, device=DEVICE, dtype=TORCH_DTYPE):
    """np配列/リスト→torch.Tensor（デバイスへ）"""
    return torch.as_tensor(x, dtype=dtype, device=device)

def _to_np(x_t: torch.Tensor):
    """torch.Tensor→np配列（CPUへ）"""
    return x_t.detach().to("cpu").numpy()

# 置換後（torch.Tensorを返す）
def torch_softmax_rank(x, lam=1.0):
    x_t = torch.as_tensor(x, dtype=TORCH_DTYPE, device=DEVICE)
    if not torch.isfinite(x_t).any():
        n = x_t.numel()
        return torch.ones(n, device=DEVICE) / n
    max_finite = torch.max(x_t[torch.isfinite(x_t)])
    z = x_t - max_finite
    y = torch.exp(z * lam)
    denom = torch.sum(y)
    if denom.item() == 0 or not torch.isfinite(denom):
        n = x_t.numel()
        return torch.ones(n, device=DEVICE) / n
    return y / denom

def torch_pick_from_probs(ids, probs):
    """
    ids: np.ndarray[int] または list[int]
    probs: np.ndarray[float] または list[float] （正規化済みでなくても可）
    戻り値: 選ばれた id (int)
    """
    ids = np.asarray(ids)
    p = np.asarray(probs, dtype=np.float64)

    # 非有限値や負値を弾きつつ正規化
    p[~np.isfinite(p)] = 0.0
    p[p < 0] = 0.0
    s = p.sum()
    if s <= 0:
        # すべて0なら一様に
        idx = np.random.randint(0, len(ids))
        return int(ids[idx])

    p = p / s

    # Torchで多項抽選（GPU/CPUどちらでも可）
    t = _to_t(p, dtype=torch.float32)
    idx_t = torch.multinomial(t, num_samples=1, replacement=True)
    idx = int(idx_t.item())
    return int(ids[idx])

# numpy 1.24+ の非推奨エイリアス対策
if not hasattr(np, "long"):
    np.long = np.int64

# ============================================================================
# GPU Batch Engine（メインループ一括化のコア）
# ============================================================================
class GPUDisplayEngine:
    def __init__(self, num_agents, num_contents, g_dim, device):
        self.device = device
        self.sim_dtype = torch.float16 if device.type == "cuda" else TORCH_DTYPE
        self.g_dim = g_dim
        self.num_agents = num_agents
        self.num_contents = num_contents

        # Agent vectors
        self.Ug = torch.zeros((num_agents, g_dim), dtype=torch.float32, device=device)
        self.Uv = torch.zeros((num_agents, g_dim), dtype=torch.float32, device=device)
        self.Ui = torch.zeros((num_agents, NUM_INSTINCT_DIM), dtype=torch.float32, device=device)

        # Content vectors
        self.Cg = None  # shape (N_contents, g_dim)
        self.Ci = None

        # pseudo history (fixed-length window)
        self.pseudo_G = torch.zeros_like(self.Ug)
        self.pseudo_I = torch.zeros_like(self.Ui)

        # ★追加：正規化済みG行列（毎ステップではなくロード時に計算）
        self.Ug_norm = None
        self.Cg_norm = None

    def load_agents(self, agents):
        UG = np.array([a.interests for a in agents], dtype=np.float32)
        UV = np.array([a.V         for a in agents], dtype=np.float32)
        UI = np.array([a.I         for a in agents], dtype=np.float32)
        self.Ug[:] = torch.from_numpy(UG).to(self.device)
        self.Uv[:] = torch.from_numpy(UV).to(self.device)
        self.Ui[:] = torch.from_numpy(UI).to(self.device)

        # ★ここを追加（like_and_dig_batch の参照名に合わせる）
        self.ug_matrix_t = self.Ug
        self.uv_matrix_t = self.Uv
        self.ui_matrix_t = self.Ui

        # ★追加：Gの正規化をここで一度だけやる
        Ug = self.Ug
        self.Ug_norm = Ug / (Ug.norm(dim=1, keepdim=True) + 1e-12)

    def load_contents(self, pool):
        CG = np.array([c.vector   for c in pool], dtype=np.float32)
        CI = np.array([c.i_vector for c in pool], dtype=np.float32)
        self.Cg = torch.from_numpy(CG).to(self.device)
        self.Ci = torch.from_numpy(CI).to(self.device)

        # ★ここを追加（like_and_dig_batch の参照名に合わせる）
        self.cg_matrix_t = self.Cg
        self.ci_matrix_t = self.Ci
        if self.sim_dtype != TORCH_DTYPE:
            self.cg_matrix_half = self.Cg.to(self.sim_dtype)
            self.ci_matrix_half = self.Ci.to(self.sim_dtype)
        else:
            self.cg_matrix_half = None
            self.ci_matrix_half = None

        # ★追加：Gの正規化をここで一度だけやる
        Cg = self.Cg
        self.Cg_norm = Cg / (Cg.norm(dim=1, keepdim=True) + 1e-12)

    # ============================================================================
    # ★ 修正版 CF（user-based / item-based 共通コア）
    # ============================================================================
    def _cf_user_based_candidates(
        self,
        isc_obj,
        agents,
        *,
        lam: float,
        face: str,
    ) -> torch.Tensor:
        """
        user_like_w / item_liked_by_w を使った user-based CF。
        - 各ユーザー u について:
            1) u が「いいね」した item 群を起点に
            2) その item を liked している他ユーザー v を辿り
            3) v が liked している他の item を、重み付きでスコア加算
        - 最終的に 未視聴 + 未like の item から ranked-softmax で1つサンプリング。
        - 戻り値: shape (num_agents,) の LongTensor（content_id）
        """
        ensure_content_index(isc_obj.pool)  # _CONTENT_IDS, _ID2ROW を更新
        global _CONTENT_IDS, _ID2ROW

        num_contents = len(isc_obj.pool)
        picked_cids: list[int] = []

        face_str = str(face).lower() if face is not None else "affinity"

        # 便利ショートカット
        user_like_w = isc_obj.user_like_w       # uid -> {ridx: base_w}
        item_liked_by_w = isc_obj.item_liked_by_w  # cid -> {uid: base_w}

        def _entry_weight(entry, container=None, key=None):
            """Convert CF like-history entry (deque or numeric) to float weight."""
            if isinstance(entry, deque):
                while entry and (step - entry[0] > CF_HISTORY_WINDOW_STEPS):
                    entry.popleft()
                if not entry:
                    if container is not None and key is not None:
                        container.pop(key, None)
                    return 0.0
                return float(len(entry))
            try:
                return float(entry)
            except (TypeError, ValueError):
                return 0.0

        for agent in agents:
            uid = int(agent.id)

            # 自分の「いいね」履歴（行インデックス ridx ベース）
            row_self = user_like_w.get(uid, {})
            if not row_self:
                # まだ like が無い場合は単純ランダム（未視聴優先）
                cid = agent.next_unseen_random_cid(num_contents)
                picked_cids.append(int(cid))
                continue

            liked_ridx_self = set(row_self.keys())

            # 候補 item のスコア（ridx ベース）
            item_scores: dict[int, float] = {}

            # u が like した各 item を起点に、v を辿る
            for ridx_i, w_ui in list(row_self.items()):
                weight_ui = _entry_weight(w_ui, row_self, ridx_i)
                if weight_ui <= 0.0:
                    continue
                # ridx -> cid -> item_liked_by_w[cid]
                cid_i = int(_CONTENT_IDS[ridx_i])
                neigh_users = item_liked_by_w.get(cid_i, {})
                if not neigh_users:
                    continue

                for v_uid, w_vi in list(neigh_users.items()):
                    v_uid = int(v_uid)
                    if v_uid == uid:
                        continue

                    weight_vi = _entry_weight(w_vi, neigh_users, v_uid)
                    if weight_vi <= 0.0:
                        continue

                    # ここでは「sim(u,v) ≒ w_ui * w_vi」を局所的に使う
                    sim_uv_local = float(weight_ui) * float(weight_vi)
                    if sim_uv_local <= 0.0:
                        continue

                    # ユーザー v の like 行（ridx）
                    row_v = user_like_w.get(v_uid, {})
                    if not row_v:
                        continue

                    for ridx_j, w_vj in list(row_v.items()):
                        weight_vj = _entry_weight(w_vj, row_v, ridx_j)
                        if weight_vj <= 0.0:
                            continue
                        # 自分が既に like した item は候補から除外
                        if ridx_j in liked_ridx_self:
                            continue
                        item_scores[ridx_j] = item_scores.get(ridx_j, 0.0) + sim_uv_local * float(weight_vj)

                    if not row_v:
                        user_like_w.pop(v_uid, None)

                if not neigh_users:
                    item_liked_by_w.pop(cid_i, None)

            # スコアが一件も出なければランダム
            if not item_scores:
                cid = agent.next_unseen_random_cid(num_contents)
                picked_cids.append(int(cid))
                continue

            # --- 未視聴フィルタ + novelty face ---
            ids: list[int] = []
            vals: list[float] = []
            seen_ids = agent.seen_content_ids

            for ridx_j, score in item_scores.items():
                cid_j = int(_CONTENT_IDS[ridx_j])
                if cid_j in seen_ids:
                    continue

                # ★ novelty face のときは「1 - cos(G_user, G_content)」で上書き
                if face_str == "novelty":
                    a_g = np.asarray(agent.interests, dtype=np.float64)
                    c_g = np.asarray(isc_obj.pool[ridx_j].vector, dtype=np.float64)

                    na = np.linalg.norm(a_g)
                    nc = np.linalg.norm(c_g)
                    if na > 0.0 and nc > 0.0:
                        cos = float(np.dot(a_g, c_g) / (na * nc))
                    else:
                        cos = 0.0  # どちらかゼロベクトルなら cos=0 扱い

                    # 「距離」＝ 1 - cos（大きいほど“新奇”）
                    score = 1.0 - cos

                vals.append(score)
                ids.append(cid_j)

            if not ids:
                cid = agent.next_unseen_random_cid(num_contents)
                picked_cids.append(int(cid))
                continue

            vals_np = np.asarray(vals, dtype=np.float64)
            # ranked-softmax（lam が温度）
            probs = softmax_arr(vals_np, lam=float(lam))
            chosen_cid = torch_pick_from_probs(ids, probs)
            picked_cids.append(int(chosen_cid))

        return torch.tensor(picked_cids, device=self.device, dtype=torch.long)

    def cf_user_affinity(self, step, isc_obj, agents):
        """
        DISPLAY_ALGORITHM == 'cf_user' 用 高速版:
        - 式や重み・確率の定義は既存の cf_user_candidates / _cf_user_based_candidates 系に完全依存。
        - ここでは「エージェントごとのキャッシュ + TTL(CF_CACHE_DURATION)」で
          計算頻度だけを落とす。
        戻り値:
            torch.LongTensor(shape=[num_agents])  … content_id
        """
        picked_ids = []
        face = CF_USER_FACE

        for a in agents:
            # ---- TTL 減算（0 で再計算）----
            if getattr(a, "cf_cache_timer", 0) > 0:
                a.cf_cache_timer -= 1

            # ---- 1) キャッシュ再利用パス ----
            cache = getattr(a, "cf_user_score_cache", [])
            if a.cf_cache_timer > 0 and cache:
                ids, probs = zip(*cache)
                ids   = np.asarray(ids,   dtype=np.int64)
                probs = np.asarray(probs, dtype=np.float64)

                # 既視聴除外 & 再正規化
                mask = np.array([cid not in a.seen_content_ids for cid in ids], dtype=bool)
                p = probs * mask
                s = p.sum()
                if np.isfinite(s) and s > 0:
                    p = p / s
                    cid = torch_pick_from_probs(ids, p)
                    picked_ids.append(int(cid))
                    # マスク後の分布でキャッシュを更新（既視聴を物理削除）
                    a.cf_user_score_cache = list(zip(ids[mask].tolist(), p.tolist()))
                    continue
                # 全部既視聴 or 不正な分布 → キャッシュ無効化して再計算へ
                a.cf_user_score_cache = []

            # ---- 2) 新規に CF 候補を計算（重い処理はここだけ）----
            a.compute_and_cache_cf_user_scores(step, agents, isc_obj, face=face)
            cache = getattr(a, "cf_user_score_cache", [])

            if cache:
                ids, probs = zip(*cache)
                ids   = np.asarray(ids,   dtype=np.int64)
                probs = np.asarray(probs, dtype=np.float64)
                cid = torch_pick_from_probs(ids, probs)
                picked_ids.append(int(cid))
            else:
                # 近傍から何も出なかった場合は未視聴ランダム
                cid = a.next_unseen_random_cid(len(isc_obj.pool))
                picked_ids.append(int(cid))

            # このエージェントの CF キャッシュ寿命をセット
            a.cf_cache_timer = CF_CACHE_DURATION

        return torch.tensor(picked_ids, device=self.device, dtype=torch.long)

    def cf_item_affinity(self, step, isc_obj, agents):
        """
        DISPLAY_ALGORITHM == 'cf_item' 用:
        - 現状は user-based CF と同じロジックに、
          CF_ITEM_FACE / LAMBDA_CF_ITEM を適用したバリエーションとして扱う。
        - 将来、本当に item-based に分岐したくなったらここを差し替えればOK。
        """
        picked_ids = []
        face = CF_ITEM_FACE

        for a in agents:
            if getattr(a, "cf_cache_timer", 0) > 0:
                a.cf_cache_timer -= 1

            cache = getattr(a, "cf_item_score_cache", [])
            if a.cf_cache_timer > 0 and cache:
                ids, probs = zip(*cache)
                ids = np.asarray(ids, dtype=np.int64)
                probs = np.asarray(probs, dtype=np.float64)
                mask = np.array([cid not in a.seen_content_ids for cid in ids], dtype=bool)
                p = probs * mask
                s = p.sum()
                if np.isfinite(s) and s > 0:
                    p = p / s
                    cid = torch_pick_from_probs(ids, p)
                    picked_ids.append(int(cid))
                    # 既視聴を除いた分布でキャッシュを更新
                    a.cf_item_score_cache = list(zip(ids[mask].tolist(), p.tolist()))
                    continue
                a.cf_item_score_cache = []

            a.compute_and_cache_cf_item_scores(step, agents, isc_obj, face=face)
            cache = getattr(a, "cf_item_score_cache", [])

            if cache:
                ids, probs = zip(*cache)
                ids = np.asarray(ids, dtype=np.int64)
                probs = np.asarray(probs, dtype=np.float64)
                cid = torch_pick_from_probs(ids, probs)
                picked_ids.append(int(cid))
            else:
                cid = a.next_unseen_random_cid(len(isc_obj.pool))
                picked_ids.append(int(cid))

            a.cf_cache_timer = CF_CACHE_DURATION

        return torch.tensor(picked_ids, device=self.device, dtype=torch.long)

    def cbf_user_affinity(self, isc_obj, agents, step, face=None):
        """
        CBF（user-based）の入口を GPUDisplayEngine 側に統一する薄いラッパ。
        実際のスコア計算ロジックは Agent.compute_and_cache_cbf_user_scores に委譲し、
        その中で既存通り isc.cbf_user_sim_matrix_t（GPU行列）が利用される。
        """
        if face is None:
            face = CBF_USER_FACE
        picked_ids = []
        pool = isc_obj.pool

        for agent in agents:
            # 既存実装そのまま利用（score 定義・既視聴除外・時間減衰などはここに全部入っている）
            content = agent.compute_and_cache_cbf_user_scores(step, agents, pool, face=face)
            # Content.id は pool 内のインデックスと一致している前提
            picked_ids.append(int(content.id))

        # メインループ側の扱いに合わせて「コンテンツIDの LongTensor」を返す
        return torch.tensor(picked_ids, dtype=torch.long, device=self.device)

    def cbf_item_affinity(self, isc_obj, agents, step, lam=30.0, face=None):
        """CBF-item（擬似ベクトル + face 対応）の GPU スコア版。

        Parameters
        ----------
        isc_obj : ISC
            コンテンツプールおよびID辞書を保持するオブジェクト。
        agents : list[UserAgent]
            対象となるエージェントのリスト（インデックス順はエージェントID順を想定）。
        step : int
            現在ステップ。擬似ベクトル計算時の時間減衰に利用。
        lam : float, optional
            ranked softmax の温度。vectorized_cbf_faced の beta にそのまま渡す。
        face : str, optional
            "affinity" / "novelty" を想定。未指定時はグローバルの CBF_ITEM_FACE を利用。
        """

        if face is None:
            face = CBF_ITEM_FACE

        ensure_content_index(isc_obj.pool)
        picked_ids = []
        global _CONTENT_IDS

        for a in agents:
            cache = getattr(a, "cbf_score_cache", [])
            # キャッシュ有効期間中は既視聴をマスクして再利用
            if getattr(isc_obj, "pseudo_cache_timer", 0) > 0 and cache:
                ids, probs = zip(*cache)
                ids_np = np.asarray(ids, dtype=np.int64)
                probs_np = np.asarray(probs, dtype=np.float64)
                mask = np.array([cid not in a.seen_content_ids for cid in ids_np], dtype=bool)
                p = probs_np * mask
                s = p.sum()
                if np.isfinite(s) and s > 0:
                    p = p / s
                    cid = torch_pick_from_probs(ids_np, p)
                    picked_ids.append(int(cid))
                    continue
                # 全部既視聴などで分布が壊れたらキャッシュを捨てて再計算
                a.cbf_score_cache = []

            # エージェントごとに擬似ベクトルを更新
            pseudo_g = a.compute_pseudo_vector_G(step)
            pseudo_i = a.compute_pseudo_vector_I(step)

            # 既視聴コンテンツは除外
            exclude_ids = a.seen_content_ids

            # 旧 cbf_item と同じロジック（GPU + 擬似 + face）でスコア＆確率を計算
            # GPU 上のコンテンツ行列全体から CBF スコアを計算
            cand_ids, probs = vectorized_cbf_faced(
                pseudo_g,
                pseudo_i,
                float(lam),        # beta (= 温度)
                face=face,
                w_g=CBF_W_G,
                w_i=CBF_W_I,
                exclude_ids=exclude_ids,
                top_k=CBF_ITEM_TOP_K,
            )
            if len(cand_ids) == 0:
                # 全候補が除外された場合のフェイルオーバー：完全ランダム
                if (_CONTENT_IDS is not None) and (len(_CONTENT_IDS) > 0):
                    ridx = random.randrange(len(_CONTENT_IDS))
                    fallback_cid = int(_CONTENT_IDS[ridx])
                else:
                    fallback_cid = int(random.randrange(max(1, len(isc_obj.pool))))
                picked_ids.append(fallback_cid)
            else:
                probs_np = np.asarray(probs, dtype=np.float64)
                ids_np = np.asarray(cand_ids, dtype=np.int64)
                probs_np[~np.isfinite(probs_np)] = 0.0
                probs_np[probs_np < 0] = 0.0
                s = probs_np.sum()
                if s <= 0:
                    chosen = int(np.random.choice(ids_np))
                else:
                    probs_np = probs_np / s
                    chosen = int(np.random.choice(ids_np, p=probs_np))
                picked_ids.append(chosen)
            # キャッシュに保存（pseudo_cache_timer 有効期間中に再利用・マスクするため）
            a.cbf_score_cache = list(zip(cand_ids.tolist(), probs.tolist()))

        # 後段処理が .cpu().numpy() 前提なので、ここでは GPU Tensor にして返す
        return torch.tensor(picked_ids, device=self.device, dtype=torch.long)

    def like_and_dig_batch(self, step, isc, agents, picked_idx_t):
        """
        GPU 上で like / dig 判定を一括計算し、
        Python 側にはフラグと補助情報だけ返す。
        """
        device = self.device
        num_agents = self.num_agents

        # [num_agents] -> [num_agents] LongTensor
        picked_idx_t = picked_idx_t.to(device=device, dtype=torch.long)

        # ---- ユーザー側埋め込み（すでに DEVICE 上にある）----
        Ug = self.ug_matrix_t          # [N, NUM_GENRES]  (Gベクトル)
        Uv = self.uv_matrix_t          # [N, NUM_GENRES]  (Vベクトル)
        Ui = self.ui_matrix_t          # [N, NUM_INSTINCT_DIM] (Iベクトル)

        # ---- コンテンツ側埋め込み（picked_idx で gather）----
        if self.cg_matrix_half is not None:
            Cg = self.cg_matrix_half[picked_idx_t].to(TORCH_DTYPE)
            Ci = self.ci_matrix_half[picked_idx_t].to(TORCH_DTYPE)
        else:
            Cg = self.cg_matrix_t[picked_idx_t]
            Ci = self.ci_matrix_t[picked_idx_t]

        eps = 1e-8
        autocast_enabled = (DEVICE.type == "cuda")
        autocast_dtype = torch.bfloat16 if CONTENT_MAT_DTYPE == "bf16" else torch.float16

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            # ---- 類似度（alpha=0: 内積, alpha>0: cos^alpha）----
            alpha_like = float(AGENT_ALPHA)

            def _sim_torch(u, v, alpha_val: float):
                # u, v : [N, D]
                dot = (u * v).sum(dim=1)
                if alpha_val == 0.0:
                    return dot
                nu = torch.linalg.norm(u, dim=1).clamp_min(eps)
                nv = torch.linalg.norm(v, dim=1).clamp_min(eps)
                denom = (nu * nv).pow(alpha_val)
                return dot / denom

            # --- like 用スコア（すべてコサイン類似度） ---
            s_cg = _sim_torch(Ug, Cg, alpha_like)
            s_cv = _sim_torch(Uv, Cg, alpha_like)
            s_ci = _sim_torch(Ui, Ci, alpha_like)

            # 線形結合（LIKE_W_* は既存のグローバル定数）
            like_score = (
                float(LIKE_W_CG) * s_cg
                + float(LIKE_W_CV) * s_cv
                + float(LIKE_W_CI) * s_ci
            )

            # ロジスティック変換 + スケーリング
            k_like = float(LOGIT_K)
            x0_like = float(LOGIT_X0)
            like_raw = torch.sigmoid(k_like * (like_score - x0_like))
            p_like = like_raw / float(LIKE_DIVISOR)

            # --- dig 用スコア（Gauss + V 依存パラメータ） ---
            # gap: C - G
            gap = Cg - Ug      # [N, NUM_GENRES]
            C = Cg
            V = Uv             # V も G と同次元

            # μ, σ, A のベクトル（既存のスカラー定数を使用）
            mu = MU0 + MU_SLOPE * (MU_ALPHA_C * C + MU_BETA_V * V)
            sigma = SIGMA0 * torch.exp(
                SIGMA_LAMDA * (SIGMA_ALPHA_C * C + SIGMA_BETA_V * V)
            )
            A = A0 * torch.exp(
                A_LAMDA * (A_ALPHA_C * C + A_BETA_V * V)
            )

            # gap>0 の次元だけ有効にする
            pos_mask = gap > 0.0

            # σ のゼロ割り防止
            sigma = sigma.clamp_min(1e-6)
            z = (gap - mu) / sigma
            gauss = A * torch.exp(-0.5 * z * z)

            # gap<=0 の次元は無効化（大きな負値に落とす）
            gauss = torch.where(pos_mask, gauss, torch.full_like(gauss, -1e9))

            # 各ユーザーごとに一番「掘りたくなる」次元を取る
            best_score, j_dim = gauss.max(dim=1)  # [N], [N]

            # ロジスティックで確率化
            k_dig = float(DIG_LOGIT_K)
            x0_dig = float(DIG_LOGIT_X0)
            dig_raw = torch.sigmoid(k_dig * (best_score - x0_dig))
            p_dig = dig_raw / float(DIG_DIVISOR)

            # --- Bernoulli サンプリング ---
            u_like = torch.rand(num_agents, device=device)
            u_dig = torch.rand(num_agents, device=device)

        like_flags_t = (u_like < p_like)
        dig_flags_t = (u_dig < p_dig)

        # CPU / numpy に戻す
        like_flags = like_flags_t.detach().cpu().numpy().astype(np.bool_)
        dig_flags = dig_flags_t.detach().cpu().numpy().astype(np.bool_)
        j_dims = j_dim.detach().cpu().numpy().astype(np.int64)
        p_like_np = p_like.detach().cpu().numpy()
        p_dig_np = p_dig.detach().cpu().numpy()

        return like_flags, dig_flags, j_dims, p_like_np, p_dig_np

    def pick_contents(self, display_algorithm, step, isc, agents):
        """
        DISPLAY_ALGORITHM ごとの「全エージェントの表示コンテンツID」を返す共通入口。

        戻り値:
            - torch.LongTensor (shape: [num_agents])  … GPUで計算したインデックス
            - または None                           … まだGPU未対応 → 旧CPU実装にフォールバック
        """
        # --- CBF: item-based（GPUバッチ版・擬似＋face対応）---
        if display_algorithm == "cbf_item":
            # INITIAL_RANDOM_STEPS や既視聴除外などは呼び出し側で制御する。
            return self.cbf_item_affinity(
                isc,
                agents,
                step,
                lam=LAMBDA_CBF_ITEM,
                face=CBF_ITEM_FACE,
            )

        # --- CBF: user-based（GPU寄せ・類似度行列は GPU を使用）---
        if display_algorithm == "cbf_user":
            # 中身のスコア計算ロジックは Agent.compute_and_cache_cbf_user_scores に委譲。
            # その内部で isc.cbf_user_sim_matrix_t (GPU) が利用されるので、
            # ここでは入口だけ統一する。
            return self.cbf_user_affinity(isc, agents, step, face=CBF_USER_FACE)

        # --- CF: user-based (GPU版) ---
        if display_algorithm == "cf_user":
            # GPUのCF行列からユーザー u に対し item候補を sampling
            return self.cf_user_affinity(step, isc, agents)

        # --- CF: item-based (GPU版) ---
        if display_algorithm == "cf_item":
            return self.cf_item_affinity(step, isc, agents)

        # --- popularity ---
        if display_algorithm == "popularity":
            # ISC 側で毎 step 更新されている GPU テンソルを参照
            global_probs = isc._GLOBAL_POP_SCORES_T  # (num_contents,) GPU
            return self._sample_from_global_probs(global_probs, agents, isc)

        # --- trend ---
        if display_algorithm == "trend":
            global_probs = isc._GLOBAL_TREND_SCORES_T
            return self._sample_from_global_probs(global_probs, agents, isc)

        # --- buzz ---
        if display_algorithm == "buzz":
            global_probs = isc._GLOBAL_BUZZ_SCORES_T
            return self._sample_from_global_probs(global_probs, agents, isc)

        # 今は cbf_item / cbf_user 以外は GPU 未対応なので None を返す
        return None

    # ---------------------------------------
    # popularity / trend / buzz 共通 GPU sampling
    # ---------------------------------------
    def _sample_from_global_probs(self, global_probs, agents, isc):
        """
        global_probs: torch.Tensor (num_contents,)  GPU
        returns: torch.LongTensor (num_agents,)     GPU

        既視聴除外だけ CPU でやり、サンプリングは GPU で行う
        """
        num_agents = len(agents)
        base_probs = global_probs.to(self.device, dtype=torch.float32)
        num_contents = base_probs.shape[0]

        # [num_agents, num_contents] をまとめて multinomial
        probs_mat = base_probs.unsqueeze(0).expand(num_agents, -1).clone()

        for a_idx, a in enumerate(agents):
            if a.seen_content_ids:
                seen_idx = torch.as_tensor(
                    list(a.seen_content_ids),
                    device=self.device,
                    dtype=torch.long,
                )
                probs_mat[a_idx].index_fill_(0, seen_idx, 0.0)

        row_sum = probs_mat.sum(dim=1)
        valid_mask = (row_sum > 0) & torch.isfinite(row_sum)

        # 正規化（無効行はそのまま）
        row_sum_clamped = row_sum.clamp_min(1e-12).view(-1, 1)
        probs_mat = probs_mat / row_sum_clamped

        picked = torch.empty(num_agents, device=self.device, dtype=torch.long)

        if valid_mask.any():
            sampled_valid = torch.multinomial(probs_mat[valid_mask], num_samples=1).squeeze(1)
            picked[valid_mask] = sampled_valid

        if (~valid_mask).any():
            picked[~valid_mask] = torch.randint(0, num_contents, size=(int((~valid_mask).sum()),), device=self.device)

        return picked

# ============================================================================
# ユーティリティ関数群
# ============================================================================
def sigmoid01(x, k=LOGIT_K, x0=LOGIT_X0):
    z = max(-60.0, min(60.0, k * (x - x0)))
    return 1.0 / (1.0 + math.exp(-z))

def softmax_arr(x, lam=1.0):
    x = np.asarray(x, dtype=np.float64)
    x = np.exp(lam * (x - np.max(x)))
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)

# --- G の多様性指標（追加） ---
def _g_entropy(vec, eps: float = 1e-12):
    """
    Gベクトル(0..1)を単純正規化して確率化し、シャノンエントロピーをlog(K)で割って0..1に正規化。
    値が高いほど“広い”（多様）興味分布。
    """
    a = np.asarray(vec, dtype=np.float64).ravel()
    K = max(1, a.size)
    s = float(a.sum())
    if s <= eps:
        return 1.0  # 全ゼロ扱い: 一様とみなして最大多様性
    p = a / s
    p = np.clip(p, eps, 1.0)
    H = -float(np.sum(p * np.log(p)))
    return H / math.log(K)

def _g_variance(vec):
    """
    Gベクトルの要素分散。値が大きいほど“凸凹（偏り）”が強い＝多様性は低い解釈。
    """
    a = np.asarray(vec, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    return float(np.var(a))

def _pearson_r(x, y):
    """
    ベクトル x, y の Pearson 相関（中心化コサイン）。
    - 長さが2未満、または片方の分散が0に近い場合は NaN を返す。
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    ax = a - a.mean()
    by = b - b.mean()
    sx = float(np.linalg.norm(ax))
    sy = float(np.linalg.norm(by))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.dot(ax, by) / (sx * sy))

def _sim_alpha(x_vec, y_vec, alpha: float):
    x = np.asarray(x_vec, dtype=np.float64).ravel()
    y = np.asarray(y_vec, dtype=np.float64).ravel()
    dot = float(np.dot(x, y))
    if alpha == 0.0:
        return dot
    nx = float(np.linalg.norm(x)); ny = float(np.linalg.norm(y))
    if nx <= 1e-9 or ny <= 1e-9:
        return 0.0
    denom = (nx * ny) ** float(alpha)
    return dot / denom

def _safe_sim_alpha(a_vec, b_vec, alpha: float):
    if a_vec is None or b_vec is None:
        return ""
    a = np.asarray(a_vec, dtype=np.float64).ravel()
    b = np.asarray(b_vec, dtype=np.float64).ravel()
    dot = float(np.dot(a, b))
    if alpha == 0.0:
        return dot
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return ""
    return dot / ((na * nb) ** float(alpha))

# --- 生成器 ---
# 切り落とし正規
def _tnorm(mu, sigma, lo, hi):
    while True:
        v = random.gauss(mu, sigma)
        if lo <= v <= hi:
            return v

def _l2(x): 
    return math.sqrt(sum(t*t for t in x)) + 1e-12

def _clip01(x: float):
    """Clamp to [0, 1]."""
    return 1.0 if x >= 1.0 else (0.0 if x <= 0.0 else x)

def _clip11(x: float):
    """Clamp to [-1, 1]."""
    return 1.0 if x >= 1.0 else (-1.0 if x <= -1.0 else x)

# --- Agent / I: 全次元アクティブ、ノルム目標にスケール ---
def _generate_agent_vector_legacy(length, mu_e, sig_e, norm_mu, norm_sig):
    # 既存の挙動そのまま（要素生成→目標ノルムへ一括スケール）
    vec = [_tnorm(mu_e, sig_e, 0.0, 1.0) for _ in range(length)]
    base = _l2(vec)
    target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
    s = target / base
    return [min(1.0, max(0.0, x * s)) for x in vec]

def generate_agent_g_vector(length, mu_e, sig_e, norm_mu, norm_sig):
    mode = str(AGENT_G_MODE).lower()
    if mode == "legacy":
        return _generate_agent_vector_legacy(length, mu_e, sig_e, norm_mu, norm_sig)
    elif mode == "element":
        vec = [_tnorm(mu_e, sig_e, 0.0, 1.0) for _ in range(length)]
        return [min(1.0, max(0.0, x)) for x in vec]
    elif mode == "norm":
        vec = [1.0] * length
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base if base > 0 else 1.0
        return [min(1.0, max(0.0, x * s)) for x in vec]
    elif mode == "random":
        return [random.random() for _ in range(length)]
    else:
        return _generate_agent_vector_legacy(length, mu_e, sig_e, norm_mu, norm_sig)

def generate_agent_i_vector(length, mu_e, sig_e, norm_mu, norm_sig):
    mode = str(AGENT_I_MODE).lower()
    if mode == "legacy":
        vec = [_tnorm(mu_e, sig_e, 0.0, 1.0) for _ in range(length)]
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base
        return [min(1.0, max(0.0, x * s)) for x in vec]
    elif mode == "element":
        return [min(1.0, max(0.0, _tnorm(mu_e, sig_e, 0.0, 1.0))) for _ in range(length)]
    elif mode == "norm":
        vec = [1.0] * length
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base if base > 0 else 1.0
        return [min(1.0, max(0.0, x * s)) for x in vec]
    elif mode == "random":
        return [random.random() for _ in range(length)]
    else:
        vec = [_tnorm(mu_e, sig_e, 0.0, 1.0) for _ in range(length)]
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base
        return [min(1.0, max(0.0, x * s)) for x in vec]

def generate_agent_v_vector(length, mu_e, sig_e, norm_mu, norm_sig):
    mode = str(AGENT_V_MODE).lower()
    if mode == "legacy":
        vec = [_tnorm(mu_e, sig_e, -1.0, 1.0) for _ in range(length)]
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base
        return [max(-1.0, min(1.0, x * s)) for x in vec]
    elif mode == "element":
        return [max(-1.0, min(1.0, _tnorm(mu_e, sig_e, -1.0, 1.0))) for _ in range(length)]
    elif mode == "norm":
        vec = [1.0] * length  # 符号は +1 で与える仕様を踏襲
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base if base > 0 else 1.0
        return [max(-1.0, min(1.0, x * s)) for x in vec]
    elif mode == "random":
        return [random.uniform(-1.0, 1.0) for _ in range(length)]
    else:
        vec = [_tnorm(mu_e, sig_e, -1.0, 1.0) for _ in range(length)]
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(length))
        s = target / base
        return [max(-1.0, min(1.0, x * s)) for x in vec]

def _generate_content_vector_unified(
    length: int, mode: str,
    mu_e: float, sig_e: float, norm_mu: float, norm_sig: float,
    active_count: int
):
    """
    mode: "element" | "norm" | "legacy" | "random"
    active_count: そのまま使う（1..kランダムではなく固定値）
    active のどの次元に立てるかはランダムサンプル（重複なし）。
    値域は 0..1 にクリップ（Vは別扱いなのでここでは想定外）。
    """
    mode = str(mode).lower()
    k = max(1, min(int(active_count), length))
    vec = np.zeros(length, dtype=np.float64)

    # active index は毎コンテンツごとにランダム選択（数は固定）
    active_idx = np.asarray(random.sample(range(length), k), dtype=np.int64)

    if mode == "element":
        for i in active_idx:
            vec[i] = _tnorm(mu_e, sig_e, 0.0, 1.0)

    elif mode == "norm":
        vec[active_idx] = 1.0
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(k))
        s = target / base if base > 0 else 1.0
        vec = np.clip(vec * s, 0.0, 1.0)

    elif mode == "legacy":
        for i in active_idx:
            vec[i] = _tnorm(mu_e, sig_e, 0.0, 1.0)
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(k))
        s = target / base if base > 0 else 1.0
        vec = np.clip(vec * s, 0.0, 1.0)

    elif mode == "random":
        for i in active_idx:
            vec[i] = random.random()

    else:
        # 未知指定は legacy と同等に
        for i in active_idx:
            vec[i] = _tnorm(mu_e, sig_e, 0.0, 1.0)
        base = _l2(vec)
        target = _tnorm(norm_mu, norm_sig, 0.0, math.sqrt(k))
        s = target / base if base > 0 else 1.0
        vec = np.clip(vec * s, 0.0, 1.0)

    return vec, active_idx

def generate_content_g_vector_unified(length, params, active_count):
    vec, active_idx = _generate_content_vector_unified(
        length=length,
        mode=CONTENT_G_MODE,
        mu_e=params["mu"], sig_e=params["sigma"],
        norm_mu=params["norm_mu"], norm_sig=params["norm_sigma"],
        active_count=active_count
    )
    return vec.astype(np.float32), active_idx

def generate_content_i_vector_unified(length, params, active_count):
    vec, active_idx = _generate_content_vector_unified(
        length=length,
        mode=CONTENT_I_MODE,
        mu_e=params["mu"], sig_e=params["sigma"],
        norm_mu=params["norm_mu"], norm_sig=params["norm_sigma"],
        active_count=active_count
    )
    return vec.astype(np.float32), active_idx

# ============================================================================
# 補充・フィルタ系ヘルパ
# ============================================================================
def _filter_unseen(ids, probs, seen_ids):
    if len(ids) == 0: return ids, probs
    m = np.array([cid not in seen_ids for cid in ids], dtype=bool)
    return ids[m], probs[m]

# 周期補充だけを採用
_REPLENISH_MAP = {}
if (REPLENISH_EVERY > 0) and (REPLENISH_COUNT > 0):
    start = max(0, int(REPLENISH_START_STEP))
    end = min(int(REPLENISH_END_STEP), MAX_STEPS)
    step_point = start
    while step_point <= end:
        _REPLENISH_MAP[step_point] = REPLENISH_COUNT
        step_point += REPLENISH_EVERY

# ============================================================================
# CBF / CF 用の行列（GとIを分離）
# ============================================================================
_CONTENT_IDS      = None
_CONTENT_G_RAW    = None   # ★追加
_CONTENT_I_RAW    = None   # ★追加
_ID2ROW           = None

def _build_content_index_from_pool(pool):
    global _CONTENT_IDS, _CONTENT_G_RAW, _CONTENT_I_RAW, _ID2ROW, _CONTENT_G_NORM, _CONTENT_I_NORM
    # CPU版（後方互換のため保持）
    g_mat = np.asarray([c.vector   for c in pool], dtype=np.float32)
    i_mat = np.asarray([c.i_vector for c in pool], dtype=np.float32)
    _CONTENT_G_RAW  = g_mat
    _CONTENT_I_RAW  = i_mat
    _CONTENT_IDS    = np.asarray([c.id for c in pool], dtype=np.int64)
    _ID2ROW         = {int(cid): int(i) for i, cid in enumerate(_CONTENT_IDS)}

    # --- GPU版（PyTorchテンソルを並行保持） ---
    # 以降のベクトル演算はこっちを使う（スコア計算を“バーン”化）
    # 行列 shape: (N_content, dim)
    globals()["_CONTENT_G_RAW_T"]  = _to_t(g_mat)                       # (N, Gdim)
    globals()["_CONTENT_I_RAW_T"]  = _to_t(i_mat)                       # (N, Idim)
    globals()["_CONTENT_G_NORM_T"] = torch.linalg.vector_norm(
        globals()["_CONTENT_G_RAW_T"], dim=1
    )                                                                    # (N,)
    globals()["_CONTENT_I_NORM_T"] = torch.linalg.vector_norm(
        globals()["_CONTENT_I_RAW_T"], dim=1
    )                                                                    # (N,)
    # 半精度(bfloat16)常駐（計算はfp32に積算）
    if DEVICE.type == "cuda":
        _DTYPE_MAT = torch.bfloat16 if CONTENT_MAT_DTYPE == "bf16" else torch.float16
    else:
        _DTYPE_MAT = torch.bfloat16
    globals()["_DTYPE_MAT"] = _DTYPE_MAT
    globals()["_CONTENT_G_RAW_T_BF16"] = globals()["_CONTENT_G_RAW_T"].to(_DTYPE_MAT).contiguous()
    globals()["_CONTENT_I_RAW_T_BF16"] = globals()["_CONTENT_I_RAW_T"].to(_DTYPE_MAT).contiguous()
    # 行ノルムはfp32のまま保持（安定性重視）

def _proj_sim_alpha_t(is_g: bool, p_vec, alpha: float) -> torch.Tensor:
    """
    行列(コンテンツ)×ベクトル(擬似)の α連続スコアを Torch で返す（GPU常駐、fp32）。
    - 行列は bfloat16 常駐、積算は fp32。
    - alpha=0 -> 内積, alpha>0 -> dot / (||row||·||p||)^alpha
    戻り値: shape (N,), dtype=float32, device=DEVICE
    """
    if p_vec is None:
        N = globals()["_CONTENT_G_RAW_T"].shape[0]
        return torch.zeros(N, device=DEVICE, dtype=torch.float32)

    M_bf16 = globals()["_CONTENT_G_RAW_T_BF16"] if is_g else globals()["_CONTENT_I_RAW_T_BF16"]
    Rn     = globals()["_CONTENT_G_NORM_T"]    if is_g else globals()["_CONTENT_I_NORM_T"]

    p_t = _to_t(p_vec).view(-1, 1)                          # fp32
    p_cast = p_t.to(M_bf16.dtype)
    dot = (M_bf16 @ p_cast).squeeze(1).to(torch.float32)

    if alpha == 0.0:
        return dot

    pn = torch.linalg.vector_norm(p_t).clamp_min(1e-12)     # fp32
    denom = (pn * Rn).pow(float(alpha))
    out = torch.zeros_like(dot)
    m = denom > 1e-12
    out[m] = dot[m] / denom[m]
    return out

def ensure_content_index(pool):
    global _CONTENT_IDS
    if pool is None or len(pool) == 0:
        _CONTENT_IDS = None
        return
    if (_CONTENT_IDS is None) or (len(_CONTENT_IDS) != len(pool)):
        _build_content_index_from_pool(pool)

def _sim_on_active_alpha(u_vec, content, alpha: float):
    idx = content.active_idx
    if idx.size == 0:
        return 0.0
    u = np.asarray(u_vec, dtype=np.float64)[idx]
    c = np.asarray(content.g_active, dtype=np.float64)
    dot = float(u @ c)
    if alpha == 0.0:
        return dot
    nu = float(np.linalg.norm(u)); nc = float(np.linalg.norm(c))
    if nu <= 1e-9 or nc <= 1e-9:
        return 0.0
    denom = (nu * nc) ** float(alpha)
    return dot / denom

# --- CBF: (wG*simG + wI*simI) の上位K ---
def vectorized_cbf_faced(
    pseudo_g: np.ndarray,
    pseudo_i: np.ndarray,
    beta: float,
    face: str,
    w_g: float,
    w_i: float,
    exclude_ids=None,
    top_k: int = 0,
):
    """
    CBF item のスコア計算（1ユーザ分）。
    - 擬似ベクトル pseudo_g / pseudo_i と
      グローバルなコンテンツ G / I 行列からスコアを計算
    - top_k>0 ならスコア上位K件に絞って ranked-softmax（0/負で無効＝全件）
    """
    global _CONTENT_IDS, _ID2ROW

    if (_CONTENT_IDS is None) or (_ID2ROW is None):
        isc_obj = globals().get("isc")
        pool = getattr(isc_obj, "pool", None) if isc_obj is not None else None
        if pool is not None:
            ensure_content_index(pool)
        else:
            raise RuntimeError("Content index is not built; ensure_content_index(pool) must be called before CBF scoring.")

    # コンテンツ行列は _proj_sim_alpha_t の中で GPU 行列を参照するので、
    # ここでは擬似ベクトルだけ渡す
    sim_g = _proj_sim_alpha_t(True,  pseudo_g, 1.0)  # (N,)
    sim_i = _proj_sim_alpha_t(False, pseudo_i, 1.0)  # (N,)

    # G/I の線形結合スコア
    score = w_g * sim_g + w_i * sim_i  # (N,)

    # Face 切り替え（affinity=そのまま / novelty=スコア反転）
    face_str = str(face).lower() if face is not None else "affinity"
    if face_str == "novelty":
        score = -score

    # 既視聴コンテンツなど除外
    if exclude_ids:
        ridxs = [_ID2ROW.get(int(cid)) for cid in exclude_ids if _ID2ROW.get(int(cid)) is not None]
        if ridxs:
            ridxs_t = torch.as_tensor(ridxs, device=DEVICE, dtype=torch.long)
            score[ridxs_t] = -float("inf")

    # ranked-softmax 用にスコアでソート
    vals, idx = torch.sort(score, descending=True)      # vals: (N,), idx: (N,)
    if top_k and top_k > 0:
        k = min(int(top_k), vals.numel())
        vals = vals[:k]
        idx = idx[:k]

    # ranked-softmax（高スコアに確率を集中させる）
    probs_t = torch_softmax_rank(vals, lam=float(beta)) # (N,)

    # コンテンツID列（ソート順）
    cids = _CONTENT_IDS[idx.detach().cpu().numpy()]     # numpy[int]
    probs_np = _to_np(probs_t).astype(np.float64)       # numpy[float]

    return cids, probs_np

# ============================================================================
# ISCクラスとコンテンツ定義
# ============================================================================
class ISC:
    class Content:
        def __init__(self, id, vector):
            self.id = id
            self.vector = np.asarray(vector, dtype=np.float32)   # G
            self.i_vector = None                                 # I
            self.views = 0
            self.likes = 0
            self.liked_by = []
            self.like_history = []

            # 最適化
            self.active_idx = np.flatnonzero(self.vector)
            self.g_active   = self.vector[self.active_idx]

            # trend/buzz
            self.trend_score = 0.0
            self.trend_ema   = 0.0
            self.prev_likes  = 0

            self.trend_score_cache = []
            self.buzz_score_cache = []

            self.g_active_count = int(content_max_active)
            self.i_active_count = int(content_i_max_active)

        def update_trend_score(self, step):
            delta_likes = self.likes - self.prev_likes
            self.trend_ema = TREND_EMA_ALPHA * delta_likes + (1 - TREND_EMA_ALPHA) * self.trend_ema
            self.trend_score = self.trend_ema
            self.prev_likes = self.likes

        def get_buzz_score(self, step):
            score = 0.0
            for step_liked, _uid in self.like_history:
                if step - step_liked <= BUZZ_WINDOW:
                    score += BUZZ_GAMMA ** (step - step_liked)
            return score

    def __init__(self, dim, content_G_PARAMS, num_contents):
        self.pool = []
        for i in range(num_contents):
            # G を4モード統一＋active数固定で生成
            g_vec, g_active_idx = generate_content_g_vector_unified(
                length=dim, params=content_G_PARAMS, active_count=content_max_active
            )
            c = self.Content(i, g_vec)

            # I も同様（NUM_INSTINCT_DIM を使用）
            i_vec, _i_active_idx = generate_content_i_vector_unified(
                length=NUM_INSTINCT_DIM, params=content_I_PARAMS, active_count=content_i_max_active
            )
            c.i_vector = i_vec

            # 最適化用キャッシュ更新（active_idx は G 基準）
            c.active_idx = np.asarray(g_active_idx, dtype=np.int64)
            c.g_active   = c.vector[c.active_idx]

            self.pool.append(c)

        self.id2content = {c.id: c for c in self.pool}
        ensure_content_index(self.pool)

        # --- ここから下（キャッシュ類）は既存のまま ---
        self.pop_cache_timer = 0
        self.trend_cache_timer = 0
        self.buzz_cache_timer = 0
        self.cbf_user_sim_matrix = None
        self.pseudo_cache_timer = 0
        self.cf_matrix_cache_timer = 0
        # CFのlike行列用: キャッシュ更新時にまとめて反映するためのステージング
        self.pending_cf_likes = []

        self.user_likes = {}  # uid -> set(content_id)
        self.item_liked_by = {}  # cid -> set(uid)
        self.cf_user_neighbors = None
        self.cf_item_neighbors = None
        # --- CF 用の遅延減衰キャッシュ ---
        self.decay_mult = 1.0
        self.last_decay_step = 0
        self.user_like_w = {}       # uid -> {ridx: base_weight}
        self.item_liked_by_w = {}   # cid -> {uid: base_weight}
        # --- CF-GPU: 行列キャッシュ ---
        self.UV_matrix = None      # User x Item の疎行列（GPU）
        self.VU_matrix = None      # Item x User の疎行列（GPU）
        self.cf_last_built_step = -1

    def display_random_to_agent(self, agent, step):
        """
        INITIAL_RANDOM_STEPS までの純ランダム表示用。
        既視聴は UserAgent 側の next_unseen_random_cid で除外される。
        """
        cid = agent.next_unseen_random_cid(len(self.pool))
        return self.id2content[int(cid)]

    def stage_cf_like(self, uid: int, cid: int, step_like: int):
        """
        CF用のlike履歴をキャッシュ更新タイミングでまとめて反映するためのステージング。
        """
        if _ID2ROW is None:
            return
        ridx = _ID2ROW.get(int(cid))
        if ridx is None:
            return
        self.pending_cf_likes.append((int(uid), int(ridx), int(step_like)))

    def replenish(self, n_new: int):
        """コンテンツプール末尾に n_new 件を補充し、索引/キャッシュを更新するだけ（挙動は後段ロジックに委ねる）"""
        if n_new <= 0:
            return
        start_id = len(self.pool)
        for k in range(n_new):
            new_id = start_id + k
            g_vec, g_active_idx = generate_content_g_vector_unified(
                length=NUM_GENRES, params=content_G_PARAMS, active_count=content_max_active
            )
            c = ISC.Content(new_id, g_vec)
            i_vec, _i_active_idx = generate_content_i_vector_unified(
                length=NUM_INSTINCT_DIM, params=content_I_PARAMS, active_count=content_i_max_active
            )
            c.i_vector = i_vec
            c.active_idx = np.asarray(g_active_idx, dtype=np.int64)
            c.g_active   = c.vector[c.active_idx]
            self.pool.append(c)

        # ID→オブジェクト辞書を更新、行列表現・norm・IDマップの再構築
        self.id2content = {c.id: c for c in self.pool}
        ensure_content_index(self.pool)
        self.pop_cache_timer = self.trend_cache_timer = self.buzz_cache_timer = 0

        # エージェント側の擬似/ランキングキャッシュも次tickで失効させる
        self.pseudo_cache_timer = 0
        self.cf_matrix_cache_timer = 0

    def tick_and_refresh_pseudo_if_needed(self, step, agents):
        need_pseudo = DISPLAY_ALGORITHM in {"cbf_item", "cbf_user"}
        need_cf = DISPLAY_ALGORITHM in {"cf_user", "cf_item"}
        if not need_pseudo and not need_cf:
            return

        if need_pseudo:
            if self.pseudo_cache_timer > 0:
                self.pseudo_cache_timer -= 1
            else:
                for a in agents:
                    a.pseudo_g_cache = None
                    a.pseudo_i_cache = None
                    a.cbf_score_cache = []
                    a.cbf_user_score_cache = []
                if DISPLAY_ALGORITHM == "cbf_user":
                    self.update_global_cbf_user_user_sims(step, agents)
                else:
                    self.cbf_user_sim_matrix = None
                self.pseudo_cache_timer = PSEUDO_CACHE_DURATION

        if not need_cf:
            return

        if self.cf_matrix_cache_timer > 0:
            self.cf_matrix_cache_timer -= 1
            return

        # pending_cf_likes を user_like_w / item_liked_by_w に反映
        if getattr(self, "pending_cf_likes", None):
            for uid, ridx, ts in self.pending_cf_likes:
                row = self.user_like_w.setdefault(int(uid), {})
                dq_u = row.setdefault(int(ridx), deque())
                dq_u.append(int(ts))

                col = self.item_liked_by_w.setdefault(int(_CONTENT_IDS[int(ridx)]), {})
                dq_i = col.setdefault(int(uid), deque())
                dq_i.append(int(ts))
            self.pending_cf_likes = []

        for a in agents:
            a.cf_score_cache = []
            a.cf_user_score_cache = []
            a.cf_item_score_cache = []
            a.cf_cache_timer = 0

        U = len(agents)
        I = len(self.pool)
        rows = []
        cols = []
        vals = []
        scale = float(self.decay_mult)
        for uid, items in list(self.user_like_w.items()):
            for ridx, dq in list(items.items()):
                while dq and (step - dq[0] > CF_HISTORY_WINDOW_STEPS):
                    dq.popleft()
                if dq:
                    rows.append(int(uid))
                    cols.append(int(ridx))
                    vals.append(float(len(dq)) * scale)
                else:
                    items.pop(ridx, None)
            if not items:
                self.user_like_w.pop(uid, None)
        if len(vals) > 0:
            rows_np = np.asarray(rows, dtype=np.int64)
            cols_np = np.asarray(cols, dtype=np.int64)
            vals_np = np.asarray(vals, dtype=np.float32)
            indices = torch.as_tensor(np.stack([rows_np, cols_np]), device=DEVICE, dtype=torch.long)
            values = torch.as_tensor(vals_np, device=DEVICE, dtype=torch.float32)
            UV = torch.sparse_coo_tensor(indices, values, size=(U, I), device=DEVICE)
            self.UV_matrix = UV.coalesce()
            self.VU_matrix = UV.transpose(0, 1).coalesce()
        else:
            self.UV_matrix = None
            self.VU_matrix = None
        self.cf_last_built_step = step

        if step > self.last_decay_step:
            self.decay_mult *= float(CF_DISCOUNT_GAMMA)
            self.last_decay_step = step

        self.cf_matrix_cache_timer = CF_CACHE_DURATION

    # ★ 追加：CF用ユーザー類似度行列の更新
    def update_global_cbf_user_user_sims(self, step, agents):
        n = len(agents)
        if n == 0:
            self.cbf_user_sim_matrix = None
            self.cbf_user_sim_matrix_t = None
            return

        # 擬似ベクトル収集（欠損ゼロ埋め）
        def _prep_stack(vecs, dim):
            X = np.zeros((len(vecs), dim), dtype=np.float32)
            for i, v in enumerate(vecs):
                if v is not None:
                    vv = np.asarray(v, dtype=np.float32).ravel()
                    X[i, :min(dim, vv.size)] = vv[:min(dim, vv.size)]
            return X

        PG = [a.compute_pseudo_vector_G(step) for a in agents]
        PI = [a.compute_pseudo_vector_I(step) for a in agents]
        Gt = _to_t(_prep_stack(PG, NUM_GENRES))            # (N,G)
        It = _to_t(_prep_stack(PI, NUM_INSTINCT_DIM))      # (N,I)

        # コサイン固定：行ノルムで正規化してから内積
        rnG = torch.linalg.vector_norm(Gt, dim=1, keepdim=True).clamp_min(1e-12)
        rnI = torch.linalg.vector_norm(It, dim=1, keepdim=True).clamp_min(1e-12)
        Gn = Gt / rnG
        In = It / rnI

        simG = (Gn @ Gn.T)     # (N,N)
        simI = (In @ In.T)
        S = float(CBF_USER_W_G) * simG + float(CBF_USER_W_I) * simI

        # 対角を -inf（自分は近傍から除外）
        diag = torch.arange(n, device=DEVICE)
        S[diag, diag] = -float("inf")

        # GPU常駐を保持（fp32）
        self.cbf_user_sim_matrix_t = S
        # 後方互換のためにCPU版も残す（必要最小限の場面でのみ使用）
        self.cbf_user_sim_matrix = _to_np(S).astype(np.float64)

    def get_content_for(self, step, agent=None, agents=None):
        if (step < INITIAL_RANDOM_STEPS) or (DISPLAY_ALGORITHM == "random") or (agent is None):
            cid = agent.next_unseen_random_cid(len(self.pool)) if agent else random.randrange(len(self.pool))
            return self.id2content[cid]

        def _sample_from_cache(pairs, seen_ids):
            if not pairs:
                return None
            ids, probs = zip(*pairs)
            ids = np.asarray(ids, dtype=np.int64)
            probs = np.asarray(probs, dtype=np.float64)
            ids, probs = _filter_unseen(ids, probs, seen_ids)
            if ids.size == 0:
                return None
            s = probs.sum()
            if (not np.isfinite(s)) or (s <= 0):
                return None
            probs = probs / s
            cid = torch_pick_from_probs(ids, probs)
            return self.id2content.get(int(cid), self.pool[0])

        alg = DISPLAY_ALGORITHM

        if alg == "popularity":
            if agent.pop_cache_timer <= 0 or not agent.pop_score_cache:
                agent.compute_and_cache_popularity_scores(step, self.pool)
            else:
                agent.pop_cache_timer -= 1
            picked = _sample_from_cache(agent.pop_score_cache, agent.seen_content_ids)
            return picked or self.id2content[agent.next_unseen_random_cid(len(self.pool))]

        if alg == "trend":
            if agent.trend_cache_timer <= 0 or not agent.trend_score_cache:
                agent.compute_and_cache_trend_scores(step, self.pool)
            else:
                agent.trend_cache_timer -= 1
            picked = _sample_from_cache(agent.trend_score_cache, agent.seen_content_ids)
            return picked or self.id2content[agent.next_unseen_random_cid(len(self.pool))]

        if alg == "buzz":
            if agent.buzz_cache_timer <= 0 or not agent.buzz_score_cache:
                agent.compute_and_cache_buzz_scores(step, self.pool)
            else:
                agent.buzz_cache_timer -= 1
            picked = _sample_from_cache(agent.buzz_score_cache, agent.seen_content_ids)
            return picked or self.id2content[agent.next_unseen_random_cid(len(self.pool))]

        if alg == "cbf_item":
            if (self.pseudo_cache_timer > 0) and agent.cbf_score_cache:
                picked = _sample_from_cache(agent.cbf_score_cache, agent.seen_content_ids)
                if picked is not None:
                    return picked
                agent.cbf_score_cache = []
            agent.compute_and_cache_cbf_scores(step, self.pool)
            picked = _sample_from_cache(agent.cbf_score_cache, agent.seen_content_ids)
            return picked or self.id2content[agent.next_unseen_random_cid(len(self.pool))]

        if alg == "cbf_user":
            return agent.compute_and_cache_cbf_user_scores(step, agents, self.pool, face=CBF_USER_FACE)

        if alg == "cf_user":
            picked_idx = engine.pick_contents("cf_user", step, self, agents)
            return self.id2content[int(picked_idx.cpu().numpy()[agent.id])]

        if alg == "cf_item":
            picked_idx = engine.pick_contents("cf_item", step, self, agents)
            return self.id2content[int(picked_idx.cpu().numpy()[agent.id])]

        raise ValueError(f"Unknown DISPLAY_ALGORITHM: {DISPLAY_ALGORITHM}")


    # --- グローバルスコア更新 ---
    def update_global_pop_scores(self):
        likes = np.array([c.likes for c in self.pool], dtype=np.float32)
        t = torch.tensor(likes, device=DEVICE)
        probs = torch_softmax_rank(t, lam=float(LAMBDA_POPULARITY))
        self._GLOBAL_POP_SCORES_T = probs  # GPU tensor

    def update_global_trend_scores_global(self):
        arr = np.array([c.trend_ema for c in self.pool], dtype=np.float32)
        t = torch.tensor(arr, device=DEVICE)
        probs = torch_softmax_rank(t, lam=float(LAMBDA_TREND))
        self._GLOBAL_TREND_SCORES_T = probs

    def update_global_buzz_scores(self, step):
        """
        buzz の“スコア計算そのもの”は従来の CPU ロジックを完全維持。
        計算後のスコア配列のみ GPU 化し softmax を取って保持する。
        """

        num_contents = len(self.pool)
        window = BUZZ_WINDOW

        # CPU 側で従来通り buzz スコアを計算
        scores = np.zeros(num_contents, dtype=np.float32)

        for i, content in enumerate(self.pool):
            total = 0.0
            # like_history: list[(step_like, uid_like)]
            for (ts_like, uid_like) in content.like_history:
                age = step - ts_like
                if age <= window:
                    total += BUZZ_GAMMA ** age
            scores[i] = total

        # ---- GPU 側テンソルに変換 ----
        scores_t = torch.tensor(scores, dtype=torch.float32, device=DEVICE)

        # ---- softmax（温度 LAMBDA_BUZZ） ----
        probs_t = torch_softmax_rank(scores_t, lam=float(LAMBDA_BUZZ))

        # ---- GPU 上に保持（Step2 方式）----
        self._GLOBAL_BUZZ_SCORES_T = probs_t

# ============================================================================
# エージェント定義（G/I擬似対応）
# ============================================================================

class UserAgent:
    def __init__(self, id):
        self.id = id
        self.interests = generate_agent_g_vector(
            NUM_GENRES,
            Agent_G_PARAMS["mu"], Agent_G_PARAMS["sigma"],
            Agent_G_PARAMS["norm_mu"], Agent_G_PARAMS["norm_sigma"]
        )
        self.V = generate_agent_v_vector(
            NUM_GENRES,
            Agent_V_PARAMS["mu"], Agent_V_PARAMS["sigma"],
            Agent_V_PARAMS["norm_mu"], Agent_V_PARAMS["norm_sigma"]
        )
        self.I = generate_agent_i_vector(
            NUM_INSTINCT_DIM,
            Agent_I_PARAMS["mu"], Agent_I_PARAMS["sigma"],
            Agent_I_PARAMS["norm_mu"], Agent_I_PARAMS["norm_sigma"]
        )
        self.initial_vector = self.interests[:]
        self.initial_V = self.V[:]
        self.initial_I = self.I[:]

        # 履歴（GとIを分離）
        self.like_history_G = []   # (step, content_G)
        self.like_history_I = []   # (step, content_I)
        self.total_likes = 0

        # ランダム巡回
        self._perm_N = NUM_CONTENTS
        self._perm_start, self._perm_stride = self._init_perm_params(self._perm_N, seed=id)
        self._perm_idx = 0

        # ログ
        self.dig_log = []
        self.impression_log = []

        self.pseudo_g_cache = None
        self.pseudo_i_cache = None

        # スコアキャッシュ
        self.cbf_score_cache = []
        self.cbf_user_score_cache = []
        self.cf_user_score_cache = []
        self.cf_item_score_cache = []
        self.cf_cache_timer = 0
        self.pop_score_cache = []
        self.pop_cache_timer = 0
        self.trend_cache_timer = 0
        self.buzz_cache_timer = 0

        # 既視聴
        self.seen_content_ids = set()

        # CF用：いいね履歴の前処理
        self.lh_steps = np.empty(0, dtype=np.int32)
        self.lh_ridx  = np.empty(0, dtype=np.int64)

        # 内積スコアログ: (step, cid, s_cg, s_cv, s_ci, liked_flag)
        self.score_log = []

    def on_pool_grew(self, new_total: int):
        """プール総数が増えたときに、巡回パラメータを新Nに適合させる"""
        new_total = int(new_total)
        if new_total <= self._perm_N:
            return
        self._perm_N = new_total
        # startはmodで収める
        self._perm_start = int(self._perm_start % new_total)
        # strideが新Nと互いに素でない場合、次の奇数へシフトして確保
        s = int(self._perm_stride)
        if s % 2 == 0:
            s += 1
        while math.gcd(s, new_total) != 1:
            s += 2
            if s >= new_total:
                s = 1
        self._perm_stride = s
        # _perm_idx は維持（巡回の続きから）

    def _init_perm_params(self, N:int, seed:int):
        rng = np.random.default_rng(RANDOM_SEED + seed)
        start = int(rng.integers(0, N))
        stride = int(rng.integers(1, min(N, 10_000))) * 2 + 1
        while math.gcd(stride, N) != 1:
            stride += 2
            if stride >= N: stride = 1
        return start, stride

    def _next_perm_cid(self):
        cid = (self._perm_start + self._perm_stride * self._perm_idx) % self._perm_N
        self._perm_idx += 1
        if self._perm_idx >= self._perm_N:
            if RANDOM_REPEAT_POLICY == "reset_when_exhausted":
                self._perm_start, self._perm_stride = self._init_perm_params(self._perm_N, seed=self.id + self._perm_idx)
                self._perm_idx = 0
            elif RANDOM_REPEAT_POLICY == "allow_when_exhausted":
                self._perm_idx = 0
            else:
                raise RuntimeError(f"Agent {self.id} has no unseen content left.")
        return int(cid)

    def next_unseen_random_cid(self, num_items):
        if int(num_items) != int(self._perm_N):
            self.on_pool_grew(int(num_items))
        for _ in range(self._perm_N):
            cid = self._next_perm_cid()
            if cid not in self.seen_content_ids:
                return cid
        # 全部見たなら仕方なく1つ返す
        return self._next_perm_cid()

    def like_prob_scores(self, uid: int, content: 'ISC.Content'):
        """
        いいね判定に使う3スコア（CG, CV, I）の“内積/連続化スコア”をそのまま返し、
        確率化は総合スコアのみ。おみくじ（winner）抽選は廃止。
        戻り値:
        p_like (float), u_random (float), s_cg (float), s_cv (float), s_ci (float)
        """
        # 3スコア（α連続：内積↔コサイン）
        s_cg = _sim_on_active_alpha(self.interests, content, AGENT_ALPHA)  # 情報(CG)
        s_cv = _sim_on_active_alpha(self.V,         content, AGENT_ALPHA)  # 好感(CV)
        s_ci = _sim_alpha(self.I, content.i_vector, AGENT_ALPHA)           # 本能(I)

        # 総合スコア → ロジスティック → スケール
        score = (LIKE_W_CG * s_cg) + (LIKE_W_CV * s_cv) + (LIKE_W_CI * s_ci)
        p_like_raw = sigmoid01(score, k=LOGIT_K, x0=LOGIT_X0)
        p_like = p_like_raw / float(LIKE_DIVISOR)

        u = random.random()
        return p_like, u, float(s_cg), float(s_cv), float(s_ci)

    def like_and_dig_scores(self, uid, content: 'ISC.Content'):
        """
        掘り判定：
        - 対象：gap_j = C[j] - G_u[j] が厳密に > 0 の次元のみ
        - パラメ化（実装に対応）:
            μ_j = MU0 + MU_SLOPE * (MU_ALPHA_C*C[j] + MU_BETA_V*V[j])
            σ_j = SIGMA0 * exp(SIGMA_LAMDA * (SIGMA_ALPHA_C*C[j] + SIGMA_BETA_V*V[j]))
            A_j = A0 * exp(A_LAMDA * (A_ALPHA_C*C[j] + A_BETA_V*V[j]))
            Gauss_j = A_j * exp(-0.5 * ((gap_j - μ_j)/σ_j)^2)
        - P_dig = σ(max_j Gauss_j; k=DIG_LOGIT_K, x0=DIG_LOGIT_X0) / DIG_DIVISOR
        """
        idx = content.active_idx
        if idx.size == 0:
            j_dim = int(np.argmax(self.interests))
            return 0.0, 0.0, j_dim

        g_u_full = np.asarray(self.interests, dtype=np.float64)
        v_full   = np.asarray(self.V, dtype=np.float64)
        c_act    = np.asarray(content.g_active, dtype=np.float64)
        idx_act  = idx

        # gap は正の次元のみ対象
        gap = c_act - g_u_full[idx_act]
        pos_mask = gap > 0.0
        if not pos_mask.any():
            j_dim = int(np.argmax(self.interests))
            return 0.0, 0.0, j_dim

        # --- Gauss 規定（μ, σ, A に α/β を別々に適用） ---
        v_act = v_full[idx_act]  # V の active 次元
        c_act = c_act            # C はそのまま（content.g_active）

        # μ(C,V) = MU0 + MU_SLOPE * ( MU_ALPHA_C*C + MU_BETA_V*V )
        mu_vec = MU0 + MU_SLOPE * (MU_ALPHA_C * c_act + MU_BETA_V * v_act)

        # σ(C,V) = SIGMA0 * exp( SIGMA_LAMDA * ( SIGMA_ALPHA_C*C + SIGMA_BETA_V*V ) )
        sigma_vec = SIGMA0 * np.exp(SIGMA_LAMDA * (SIGMA_ALPHA_C * c_act + SIGMA_BETA_V * v_act))

        # A(C,V) = A0 * exp( A_LAMDA * ( A_ALPHA_C*C + A_BETA_V*V ) )
        A_vec = A0 * np.exp(A_LAMDA * (A_ALPHA_C * c_act + A_BETA_V * v_act))

        # --- ガウススコア ---
        z = (gap - mu_vec) / sigma_vec
        gauss = A_vec * np.exp(-0.5 * z * z)

        score_vec = gauss
        score_vec[~pos_mask] = -np.inf

        j_local = int(np.argmax(score_vec))
        best_score = float(score_vec[j_local])
        j_dim = int(idx_act[j_local])

        if not np.isfinite(best_score):
            return 0.0, 0.0, j_dim

        p_dig_raw = sigmoid01(best_score, k=float(DIG_LOGIT_K), x0=float(DIG_LOGIT_X0))
        p_dig = p_dig_raw / float(DIG_DIVISOR)
        return 0.0, max(0.0, min(1.0, p_dig)), j_dim

    # --- == ②：擬似（G/I） == ---
    def update_like_history_GI(self, g_vec, i_vec, step):
        self.like_history_G.append((step, g_vec))
        self.like_history_I.append((step, i_vec))

    def _compute_pseudo(self, history, current_step, dim_hint=None):
        if not history:
            return None
        cutoff = int(PSEUDO_HISTORY_WINDOW_STEPS)
        valid_vecs = []
        weights = []

        for s, vec in history:
            if vec is None:
                continue
            age = current_step - int(s)
            if age < 0 or age > cutoff:
                continue
            arr = np.asarray(vec, dtype=np.float64).ravel()
            if arr.size == 0:
                continue
            valid_vecs.append(arr)
            weights.append(PSEUDO_DISCOUNT_GAMMA ** age)

        if not weights:
            return None
        stacked = np.stack(valid_vecs, axis=0)
        w = np.asarray(weights, dtype=np.float64)
        wsum = float(w.sum())
        if (not np.isfinite(wsum)) or (wsum <= 0):
            return None
        weighted = (w[:, None] * stacked).sum(axis=0) / wsum
        return weighted.tolist()

    # --- CBF ---
    def compute_and_cache_cbf_scores(self, step, content_pool):
        ensure_content_index(content_pool)
        pg = self.compute_pseudo_vector_G(step)
        pi = self.compute_pseudo_vector_I(step)
        cids, probs = vectorized_cbf_faced(
            pg,
            pi,
            LAMBDA_CBF_ITEM,
            face=CBF_ITEM_FACE,
            w_g=CBF_W_G,
            w_i=CBF_W_I,
            exclude_ids=self.seen_content_ids,
            top_k=CBF_ITEM_TOP_K,
        )
        self.cbf_score_cache = list(zip(cids.tolist(), probs.tolist()))

    # --- CF ---
    def compute_and_cache_cbf_user_scores(self, step, agents, content_pool, face):
        cached = self._cbf_user_try_cache_hit(content_pool)
        if cached is not None:
            return cached

        ensure_content_index(content_pool)
        pseudo = self._cbf_user_pseudo_vectors(step)
        if pseudo is None:
            self.cbf_user_score_cache = []
            return random.choice(content_pool)
        pG, pI = pseudo  # 参照用（現状は有無判定のみ）

        isc_obj = globals().get("isc")
        neigh_idx, neigh_sims = self._cbf_user_prepare_neighbors(step, agents, isc_obj)
        ids_arr, score_array, denom_array, has_any = self._cbf_user_collect_scores(
            step, agents, content_pool, neigh_idx, neigh_sims
        )
        if not has_any:
            self.cbf_user_score_cache = []
            return random.choice(content_pool)

        keep_idx, probs = self._cbf_user_finalize_distribution(
            ids_arr, score_array, denom_array, face
        )
        if keep_idx is None:
            return random.choice(content_pool)

        cid = torch_pick_from_probs(ids_arr[keep_idx], probs)
        return next((c for c in content_pool if c.id == cid), random.choice(content_pool))

    def _cbf_user_try_cache_hit(self, content_pool):
        isc_obj = globals().get("isc")
        if (isc_obj is None) or (isc_obj.pseudo_cache_timer <= 0) or not self.cbf_user_score_cache:
            return None
        ids, probs = zip(*self.cbf_user_score_cache)
        ids = np.asarray(ids, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float64)
        mask = np.array([cid not in self.seen_content_ids for cid in ids], dtype=bool)
        p = probs * mask
        s = p.sum()
        if np.isfinite(s) and s > 0:
            p = p / s
            cid = int(np.random.choice(ids, p=p))
            return next((c for c in content_pool if c.id == cid), random.choice(content_pool))
        self.cbf_user_score_cache = []
        return None

    def _cbf_user_pseudo_vectors(self, step):
        pG = self.compute_pseudo_vector_G(step)
        pI = self.compute_pseudo_vector_I(step)
        has_g = (pG is not None) and (np.linalg.norm(pG) > 0)
        has_i = (pI is not None) and (np.linalg.norm(pI) > 0)
        if not (has_g or has_i):
            return None
        return (pG if has_g else None, pI if has_i else None)

    def _cbf_user_prepare_neighbors(self, step, agents, isc_obj):
        if isc_obj.cbf_user_sim_matrix is None:
            isc_obj.update_global_cbf_user_user_sims(step, agents)
        if getattr(isc_obj, "cbf_user_sim_matrix_t", None) is not None:
            sims_row_t = isc_obj.cbf_user_sim_matrix_t[self.id, :]
            K = min(CBF_USER_TOP_K_USERS, len(agents) - 1)
            vals_t, idx_t = torch.topk(sims_row_t, K, largest=True, sorted=True)
            return idx_t.to("cpu").numpy(), vals_t.to("cpu").numpy()
        sims_row = isc_obj.cbf_user_sim_matrix[self.id]
        K = min(CBF_USER_TOP_K_USERS, len(agents) - 1)
        neigh_idx = np.argpartition(-sims_row, K - 1)[:K]
        neigh_idx = neigh_idx[np.argsort(-sims_row[neigh_idx])]
        neigh_sims = sims_row[neigh_idx]
        return neigh_idx, neigh_sims

    def _cbf_user_collect_scores(self, step, agents, content_pool, neigh_idx, neigh_sims):
        ids_arr = np.asarray([c.id for c in content_pool], dtype=np.int64)
        num_items = len(content_pool)
        score_tensor = torch.zeros(num_items, dtype=torch.float32, device=DEVICE)
        denom_tensor = torch.zeros(num_items, dtype=torch.float32, device=DEVICE)
        win = int(CBF_USER_HISTORY_WINDOW_STEPS)
        gamma = torch.tensor(float(CBF_USER_DISCOUNT_GAMMA), dtype=torch.float32, device=DEVICE)
        seen_idx_arr = None
        if self.seen_content_ids:
            ridx_seen = [_ID2ROW.get(int(cid)) for cid in self.seen_content_ids if _ID2ROW.get(int(cid)) is not None]
            if ridx_seen:
                seen_idx_arr = np.asarray(ridx_seen, dtype=np.int64)

        def accumulate(max_age_steps):
            idx_buf = []
            num_buf = []
            den_buf = []
            for sim, uid in zip(neigh_sims, neigh_idx):
                if not (sim > 0 or sim < 0):
                    continue
                other = agents[int(uid)]
                if other.lh_steps.size == 0:
                    continue
                m = (step - other.lh_steps) <= max_age_steps
                if not m.any():
                    continue
                idxs_all = other.lh_ridx[m]
                ages_all = (step - other.lh_steps[m]).astype(np.int64)
                if seen_idx_arr is not None and idxs_all.size:
                    keep = ~np.isin(idxs_all, seen_idx_arr)
                    if not keep.any():
                        continue
                    idxs = idxs_all[keep]
                    ages = ages_all[keep]
                else:
                    idxs = idxs_all
                    ages = ages_all
                if idxs.size == 0:
                    continue
                idxs_t = torch.as_tensor(idxs, dtype=torch.long, device=DEVICE)
                ages_t = torch.as_tensor(ages, dtype=torch.float32, device=DEVICE)
                weights = torch.pow(gamma, ages_t)
                sim_val = float(sim)
                num_buf.append(weights * sim_val)
                den_buf.append(weights * abs(sim_val))
                idx_buf.append(idxs_t)
            if idx_buf:
                idxs_cat = torch.cat(idx_buf)
                num_cat = torch.cat(num_buf)
                den_cat = torch.cat(den_buf)
                score_tensor.index_add_(0, idxs_cat, num_cat)
                denom_tensor.index_add_(0, idxs_cat, den_cat)

        def refresh_arrays():
            return (
                score_tensor.detach().cpu().numpy(),
                denom_tensor.detach().cpu().numpy(),
            )

        def count_candidates(denom_array):
            unseen_mask = np.array([cid not in self.seen_content_ids for cid in ids_arr], dtype=bool)
            return int(np.count_nonzero((denom_array > 0) & unseen_mask))

        accumulate(win)
        score_array, denom_array = refresh_arrays()
        tries = 0
        while (count_candidates(denom_array) < CBF_USER_TARGET_MIN_CANDIDATES) and (
            tries < CBF_USER_BACKFILL_MAX_RETRIES
        ):
            win = int(max(win, 1) * max(CBF_USER_BACKFILL_GROWTH, 1.0))
            accumulate(win)
            score_array, denom_array = refresh_arrays()
            tries += 1

        has_any = count_candidates(denom_array) > 0
        return ids_arr, score_array, denom_array, has_any

        def _cbf_user_finalize_distribution(self, ids_arr, score_array, denom_array, face):
            score_norm = np.full_like(score_array, -np.inf, dtype=np.float32)
            m_valid = denom_array > 0
            score_norm[m_valid] = score_array[m_valid] / denom_array[m_valid]

            if self.seen_content_ids:
                ridx = [_ID2ROW.get(cid) for cid in self.seen_content_ids if _ID2ROW.get(cid) is not None]
                if ridx:
                    ridx = np.asarray(ridx, dtype=np.int64)
                    score_norm[ridx] = -np.inf

            keep_mask = np.isfinite(score_norm) & np.array(
                [cid not in self.seen_content_ids for cid in ids_arr], dtype=bool
            )
            if not keep_mask.any():
                # ランダムフォールバックをキャッシュ
                cid = self.next_unseen_random_cid(len(ids_arr))
                self.cbf_user_score_cache = [(int(cid), 1.0)]
                return np.asarray([int(cid)], dtype=np.int64), np.asarray([1.0], dtype=np.float64)

            keep_idx = np.where(keep_mask)[0]
            try:
                cap = int(CBF_USER_CONTENT_TOP_K)
                if keep_idx.size > cap:
                    part = np.argpartition(-score_norm[keep_idx], cap - 1)[:cap]
                    keep_idx = keep_idx[part]
            except Exception:
                pass

        score_slice = score_norm[keep_idx]
        face_str = str(face).lower() if face is not None else "affinity"
        if face_str == "novelty":
            score_slice = -score_slice
        probs = softmax_arr(score_slice, LAMBDA_CBF_USER)
        self.cbf_user_score_cache = list(zip(ids_arr[keep_idx].tolist(), probs.tolist()))
        return keep_idx, probs

    def compute_and_cache_cf_user_scores(self, step, agents, isc_obj, face):
        ids, probs = cf_user_candidates(isc_obj, self.id, step, agents, face=face)
        if ids.size == 0:
            # ランダムフォールバックをキャッシュ
            cid = self.next_unseen_random_cid(len(isc_obj.pool))
            self.cf_user_score_cache = [(int(cid), 1.0)]
        else:
            self.cf_user_score_cache = list(zip(ids.tolist(), probs.tolist()))

    def compute_and_cache_cf_item_scores(self, step, agents, isc_obj, face):
        ids, probs = cf_item_candidates(isc_obj, self.id, step, agents, face=face)
        if ids.size == 0:
            cid = self.next_unseen_random_cid(len(isc_obj.pool))
            self.cf_item_score_cache = [(int(cid), 1.0)]
        else:
            self.cf_item_score_cache = list(zip(ids.tolist(), probs.tolist()))


    # UserAgent クラス末尾あたりに追加（メソッド）
    def compute_pseudo_vector_G(self, step):
        if self.pseudo_g_cache is None:
            self.pseudo_g_cache = self._compute_pseudo(self.like_history_G, step, dim_hint=NUM_GENRES)
        return self.pseudo_g_cache

    def compute_pseudo_vector_I(self, step):
        if self.pseudo_i_cache is None:
            self.pseudo_i_cache = self._compute_pseudo(self.like_history_I, step, dim_hint=NUM_INSTINCT_DIM)
        return self.pseudo_i_cache

# ============================================================================
# CF（user/item）候補生成ヘルパ
# ============================================================================

def _cf_candidates_core(isc_obj, target_agent, step: int, *, lam: float, face: str):
    """
    GPU 行列が利用できれば高速パス、それ以外は CPU 実装。
    """
    if DEVICE.type != "cpu":
        gpu_res = _cf_candidates_core_gpu(isc_obj, target_agent, step=step, lam=lam, face=face)
        if gpu_res is not None:
            return gpu_res
    res = _cf_candidates_core_cpu(isc_obj, target_agent, step=step, lam=lam, face=face)
    # 空ならランダムフォールバック
    if res[0].size == 0:
        cid = target_agent.next_unseen_random_cid(len(isc_obj.pool))
        return (np.asarray([cid], dtype=np.int64), np.asarray([1.0], dtype=np.float64))
    return res


def _cf_candidates_core_cpu(isc_obj, target_agent, step: int, *, lam: float, face: str):
    """
    1ユーザー分の CF 候補を計算して、(content_ids, probs) を返す共通コア。
    - isc_obj.user_like_w / item_liked_by_w を使った user-based CF
    - 既視聴 / 既Likeは除外
    - face == "novelty" のときは「1 - cos(G_user, G_content)」でスコア上書き
    """
    ensure_content_index(isc_obj.pool)
    global _CONTENT_IDS, _ID2ROW

    uid = int(target_agent.id)
    seen_ids = target_agent.seen_content_ids
    face_str = str(face).lower() if face is not None else "affinity"

    user_like_w = isc_obj.user_like_w       # uid -> {ridx: deque}
    item_liked_by_w = isc_obj.item_liked_by_w  # cid -> {uid: deque}

    # 自分の like 行（ridx ベース）
    row_self = user_like_w.get(uid, {})
    if not row_self:
        # いいね履歴がない場合は空を返す（呼び出し側でランダムフォールバック）
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))

    liked_ridx_self = set(row_self.keys())
    item_scores: dict[int, float] = {}

    # --- 近傍ユーザー経由で item スコアを集計 ---
    for ridx_i, dq_ui in list(row_self.items()):
        while dq_ui and (step - dq_ui[0] > CF_HISTORY_WINDOW_STEPS):
            dq_ui.popleft()
        if not dq_ui:
            row_self.pop(ridx_i, None)
            continue
        w_ui = len(dq_ui)
        cid_i = int(_CONTENT_IDS[ridx_i])
        neigh_users = item_liked_by_w.get(cid_i, {})
        if not neigh_users:
            continue
        neigh_items = list(neigh_users.items())
        if CF_NEIGHBOR_TOP_K > 0 and len(neigh_items) > CF_NEIGHBOR_TOP_K:
            neigh_items = sorted(neigh_items, key=lambda kv: len(kv[1]), reverse=True)[:CF_NEIGHBOR_TOP_K]

        for v_uid, dq_vi in neigh_items:
            v_uid = int(v_uid)
            if v_uid == uid:
                continue
            while dq_vi and (step - dq_vi[0] > CF_HISTORY_WINDOW_STEPS):
                dq_vi.popleft()
            if not dq_vi:
                neigh_users.pop(v_uid, None)
                continue

            sim_uv_local = float(w_ui) * float(len(dq_vi))
            if sim_uv_local <= 0.0:
                continue

            row_v = user_like_w.get(v_uid, {})
            if not row_v:
                continue

            for ridx_j, dq_vj in list(row_v.items()):
                while dq_vj and (step - dq_vj[0] > CF_HISTORY_WINDOW_STEPS):
                    dq_vj.popleft()
                if not dq_vj:
                    row_v.pop(ridx_j, None)
                    continue
                # 自分が既に like した item は候補から除外
                if ridx_j in liked_ridx_self:
                    continue
                item_scores[ridx_j] = item_scores.get(ridx_j, 0.0) + sim_uv_local * float(len(dq_vj))

    if not item_scores:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))

    # --- 既視聴除外 + face (affinity / novelty) 適用 ---
    ids: list[int] = []
    vals: list[float] = []

    a_g = np.asarray(target_agent.interests, dtype=np.float64)

    for ridx_j, base_score in item_scores.items():
        cid_j = int(_CONTENT_IDS[ridx_j])
        # 既視聴は弾く
        if cid_j in seen_ids:
            continue

        score = float(base_score)

        if face_str == "novelty":
            c_g = np.asarray(isc_obj.pool[ridx_j].vector, dtype=np.float64)
            na = float(np.linalg.norm(a_g))
            nc = float(np.linalg.norm(c_g))
            if na > 0.0 and nc > 0.0:
                cos = float(np.dot(a_g, c_g) / (na * nc))
            else:
                cos = 0.0
            # 大きいほど“新奇”になるよう 1 - cos
            score = 1.0 - cos

        ids.append(cid_j)
        vals.append(score)

    if not ids:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))

    ids_np = np.asarray(ids, dtype=np.int64)
    vals_np = np.asarray(vals, dtype=np.float64)

    if CF_CANDIDATE_TOP_K > 0 and vals_np.size > CF_CANDIDATE_TOP_K:
        part = np.argpartition(-vals_np, CF_CANDIDATE_TOP_K - 1)[:CF_CANDIDATE_TOP_K]
        order = part[np.argsort(-vals_np[part])]
        vals_np = vals_np[order]
        ids_np = ids_np[order]

    # ranked-softmax で確率化（温度 lam）
    probs = softmax_arr(vals_np, lam=float(lam))
    return ids_np, probs


def cf_user_candidates(isc_obj, uid: int, step: int, agents, *, face: str = "affinity"):
    """
    1ユーザー(uid)向けの CF(user-based) 候補分布を返す。
    戻り値:
        ids  : np.ndarray[int]  （content_id 群）
        probs: np.ndarray[float]（対応する確率）
    """
    # agents[uid] が id と1対1で対応している前提（現状の生成ロジックと一致）
    target_agent = agents[int(uid)]
    return _cf_candidates_core(
        isc_obj,
        target_agent,
        step,
        lam=LAMBDA_CF_USER,
        face=face,
    )


def cf_item_candidates(isc_obj, uid: int, step: int, agents, *, face: str = "affinity"):
    """
    現状の仕様では、CF(item-based) も user-based と同じロジックを使う。
    将来的に本当の item-based にしたくなったらここを差し替えればOK。

    戻り値:
        ids  : np.ndarray[int]  （content_id 群）
        probs: np.ndarray[float]（対応する確率）
    """
    target_agent = agents[int(uid)]
    return _cf_candidates_core(
        isc_obj,
        target_agent,
        step,
        lam=LAMBDA_CF_ITEM,
        face=face,
    )


def _cf_candidates_core_gpu(isc_obj, target_agent, step: int, *, lam: float, face: str):
    global _CONTENT_IDS, _ID2ROW
    ensure_content_index(isc_obj.pool)
    if (_CONTENT_IDS is None) or (len(_CONTENT_IDS) == 0):
        return None

    uid = int(target_agent.id)
    seen_ids = target_agent.seen_content_ids
    seen_ridx_arr = None
    if seen_ids:
        ridx_seen = [_ID2ROW.get(int(cid)) for cid in seen_ids if _ID2ROW.get(int(cid)) is not None]
        if ridx_seen:
            seen_ridx_arr = np.asarray(ridx_seen, dtype=np.int64)
    user_like_w = isc_obj.user_like_w
    item_liked_by_w = isc_obj.item_liked_by_w
    row_self = user_like_w.get(uid, {})
    if not row_self:
        return None

    neighbor_sim = {}
    K_neigh = int(CF_NEIGHBOR_TOP_K)
    liked_ridx_self = set(row_self.keys())
    for ridx_i, dq_ui in list(row_self.items()):
        while dq_ui and (step - dq_ui[0] > CF_HISTORY_WINDOW_STEPS):
            dq_ui.popleft()
        if not dq_ui:
            row_self.pop(ridx_i, None)
            continue
        w_ui = len(dq_ui)
        cid_i = int(_CONTENT_IDS[ridx_i])
        neigh_users = item_liked_by_w.get(cid_i, {})
        if not neigh_users:
            continue
        for v_uid, dq_vi in list(neigh_users.items()):
            v_uid = int(v_uid)
            if v_uid == uid:
                continue
            while dq_vi and (step - dq_vi[0] > CF_HISTORY_WINDOW_STEPS):
                dq_vi.popleft()
            if not dq_vi:
                neigh_users.pop(v_uid, None)
                continue
            sim_uv = float(w_ui) * float(len(dq_vi))
            if sim_uv <= 0.0:
                continue
            neighbor_sim[v_uid] = neighbor_sim.get(v_uid, 0.0) + sim_uv

    if not neighbor_sim:
        return None

    # 近傍Top-Kで絞り込み（K<=0なら全件）
    if K_neigh > 0 and len(neighbor_sim) > K_neigh:
        # neighbor_sim は dict: uid -> sim。上位Kのみ残す
        top_items = sorted(neighbor_sim.items(), key=lambda kv: kv[1], reverse=True)[:K_neigh]
        neighbor_sim = {uid: sim for uid, sim in top_items}

    K_cand = int(CF_CANDIDATE_TOP_K)
    rows = []
    vals = []
    for v_uid, sim_val in neighbor_sim.items():
        row_v = user_like_w.get(v_uid, {})
        if not row_v:
            continue
        for ridx_j, dq_vj in list(row_v.items()):
            while dq_vj and (step - dq_vj[0] > CF_HISTORY_WINDOW_STEPS):
                dq_vj.popleft()
            if not dq_vj:
                row_v.pop(ridx_j, None)
                continue
            if ridx_j in liked_ridx_self:
                continue
            weight = float(len(dq_vj))
            if weight <= 0.0:
                continue
            rows.append(int(ridx_j))
            vals.append(float(sim_val) * weight)

    if not rows:
        cid = target_agent.next_unseen_random_cid(len(isc_obj.pool))
        return (np.asarray([cid], dtype=np.int64), np.asarray([1.0], dtype=np.float64))

    scores_t = torch.zeros(len(_CONTENT_IDS), dtype=torch.float32, device=DEVICE)
    idxs_t = torch.tensor(rows, dtype=torch.long, device=DEVICE)
    vals_t = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
    scores_t.index_add_(0, idxs_t, vals_t)
    scores = scores_t.detach().cpu().numpy()

    mask = np.ones_like(scores, dtype=bool)
    if seen_ridx_arr is not None:
        mask[seen_ridx_arr] = False

    finite = np.isfinite(scores) & mask
    if not finite.any():
        cid = target_agent.next_unseen_random_cid(len(isc_obj.pool))
        return (np.asarray([cid], dtype=np.int64), np.asarray([1.0], dtype=np.float64))

    face_str = str(face).lower() if face is not None else "affinity"
    if face_str == "novelty":
        for ridx in np.where(finite)[0]:
            content = isc_obj.pool[int(ridx)]
            cos = _safe_sim_alpha(target_agent.interests, content.vector, 1.0)
            if cos == "":
                cos = 0.0
            scores[ridx] = 1.0 - float(cos)

    vals = scores[finite]
    ids = _CONTENT_IDS[finite]
    if K_cand > 0 and vals.size > K_cand:
        part_idx = np.argpartition(-vals, K_cand - 1)[:K_cand]
        keep = part_idx[np.argsort(-vals[part_idx])]
        vals = vals[keep]
        ids = ids[keep]
    probs = softmax_arr(vals, lam=float(lam))
    # safety: 万一空になった場合
    if ids.size == 0:
        cid = target_agent.next_unseen_random_cid(len(isc_obj.pool))
        return (np.asarray([cid], dtype=np.int64), np.asarray([1.0], dtype=np.float64))
    return ids, probs


def _pick_ids_via_cpu_algorithm(isc_obj, agents, step):
    """
    DISPLAY_ALGORITHM が cbf_user / cf_item のように CPU 実装のみ存在する場合に、
    既存の ISC.get_content_for をそのまま呼び出して content.id の配列を返すヘルパ。
    """
    picked_ids = np.empty(len(agents), dtype=np.int64)
    for idx, agent in enumerate(agents):
        content = isc_obj.get_content_for(step, agent=agent, agents=agents)
        cid = getattr(content, "id", None)
        if cid is None:
            cid = agent.next_unseen_random_cid(len(isc_obj.pool))
        picked_ids[idx] = int(cid)
    return torch.from_numpy(picked_ids).to(DEVICE)

def _random_interval_active(step: int) -> bool:
    """初期ランダム終了後にランダムウィンドウを挿入するか判定"""
    if not RANDOM_INTERVAL_ON:
        return False
    if RANDOM_RANDOM_BLOCK_LEN <= 0:
        return False
    cycle = RANDOM_RANDOM_BLOCK_LEN + max(0, RANDOM_NORMAL_BLOCK_LEN)
    if cycle <= 0:
        return False
    if step < INITIAL_RANDOM_STEPS:
        return True  # 初期ランダム期間は別扱い
    rel = step - INITIAL_RANDOM_STEPS
    return (rel % cycle) < RANDOM_RANDOM_BLOCK_LEN

# ============================================================================
# 実行セットアップ
# ============================================================================

# コンテンツとエージェント
isc = ISC(NUM_GENRES, content_G_PARAMS, NUM_CONTENTS)
agents = [UserAgent(i) for i in range(NUM_AGENTS)]

# ============================================================================
# メインループ（GPU batch）
# ============================================================================
engine = GPUDisplayEngine(NUM_AGENTS, NUM_CONTENTS, NUM_GENRES, DEVICE)
engine.load_agents(agents)
engine.load_contents(isc.pool)

t0 = time.time()
for step in range(MAX_STEPS):

    isc.tick_and_refresh_pseudo_if_needed(step, agents)
    # ---- グローバルスコア(popularity / trend / buzz) の更新 ----
    # popularity
    if DISPLAY_ALGORITHM == "popularity":
        if isc.pop_cache_timer <= 0:
            isc.update_global_pop_scores()
            isc.pop_cache_timer = POP_CACHE_DURATION
        else:
            isc.pop_cache_timer -= 1

    # trend
    if DISPLAY_ALGORITHM == "trend":
        if isc.trend_cache_timer <= 0:
            # 各コンテンツの trend_ema / trend_score を更新してから GPU テンソル化
            for c in isc.pool:
                c.update_trend_score(step)
            isc.update_global_trend_scores_global()
            isc.trend_cache_timer = TREND_CACHE_DURATION
        else:
            isc.trend_cache_timer -= 1

    # buzz
    if DISPLAY_ALGORITHM == "buzz":
        if isc.buzz_cache_timer <= 0:
            isc.update_global_buzz_scores(step)
            isc.buzz_cache_timer = BUZZ_CACHE_DURATION
        else:
            isc.buzz_cache_timer -= 1

    # ---- 補充（このロジックはそのまま維持）----
    if step in _REPLENISH_MAP:
        isc.replenish(_REPLENISH_MAP[step])
        engine.load_contents(isc.pool)
        NUM_CONTENTS = len(isc.pool)

    # ---- 候補コンテンツ選択 ----
    force_random = (step < INITIAL_RANDOM_STEPS) or _random_interval_active(step)

    if DISPLAY_ALGORITHM == "random" or force_random:
        # 完全ランダム表示（未視聴から選ぶ）
        picked_idx_cpu = np.zeros(NUM_AGENTS, dtype=np.int64)
        for a_id, agent in enumerate(agents):
            content = isc.display_random_to_agent(agent, step)
            picked_idx_cpu[a_id] = int(content.id)
        picked_idx = torch.from_numpy(picked_idx_cpu).to(DEVICE, dtype=torch.long)

    else:
        # random 以外のアルゴリズム（通常経路）
        # GPU バッチで候補選択
        picked_idx = engine.pick_contents(
            DISPLAY_ALGORITHM,
            step,
            isc,
            agents,
        )
        # GPU 未対応アルゴリズム（cf_user / cf_item 等）は CPU フォールバック
        if picked_idx is None:
            picked_idx = _pick_ids_via_cpu_algorithm(isc, agents, step)

    # ---- GPU で like / dig 判定を一括実行 ----
    like_flags, dig_flags, j_dims, p_like_arr, p_dig_arr = engine.like_and_dig_batch(
        step,
        isc,
        agents,
        picked_idx,
    )

    # ---- CPU 側オブジェクトへの反映（bookkeeping のみ）----
    picked_idx_cpu = picked_idx.detach().cpu().numpy()

    g_updates = []
    v_updates = []

    for a_id, cid in enumerate(picked_idx_cpu):
        content = isc.pool[int(cid)]
        agent = agents[a_id]

        # 閲覧カウンタ
        content.views += 1
        agent.seen_content_ids.add(content.id)

        # ============================================================================
        # ★ ログ: score_log / impression_log / dig_log
        # ============================================================================

        # --- いいね用スコア（CG, CV, CI）を CPU 側で再計算してログに積む ---
        s_cg = _sim_on_active_alpha(agent.interests, content, AGENT_ALPHA)  # 情報(CG)
        s_cv = _sim_on_active_alpha(agent.V,         content, AGENT_ALPHA)  # 好感(CV)
        s_ci = _sim_alpha(agent.I, content.i_vector, AGENT_ALPHA)           # 本能(I)

        # score_log: (step, cid, s_cg, s_cv, s_ci, liked_flag)
        liked_flag_int = 1 if bool(like_flags[a_id]) else 0
        agent.score_log.append(
            (int(step), int(content.id), float(s_cg), float(s_cv), float(s_ci), liked_flag_int)
        )

        # --- impression_log: (step, cid, like_flag, cos, pearson_r_g_content, pearson_r_v_content) ---
        # cos は user G と content G のコサイン類似度
        cos_g = _safe_sim_alpha(agent.interests, content.vector, 1.0)

        # Pearson は「active 次元」の G に対して計算（以前の仕様に近づける）
        if content.active_idx.size > 0:
            idx_act = content.active_idx
            g_user_act = np.asarray(agent.interests, dtype=np.float64)[idx_act]
            g_cont_act = np.asarray(content.vector,   dtype=np.float64)[idx_act]
            pearson_g = _pearson_r(g_user_act, g_cont_act)

            v_user_act = np.asarray(agent.V, dtype=np.float64)[idx_act]
            pearson_v = _pearson_r(v_user_act, g_cont_act)
        else:
            pearson_g = float("nan")
            pearson_v = float("nan")

        agent.impression_log.append(
            (int(step), int(content.id), liked_flag_int, float(cos_g), float(pearson_g), float(pearson_v))
        )

        if bool(dig_flags[a_id]):
            j_dim = int(j_dims[a_id])
            # その次元のコンテンツG値（参考用）
            if 0 <= j_dim < len(content.vector):
                g_on_j = float(content.vector[j_dim])
            else:
                g_on_j = 0.0

            # rank_g: コンテンツGの次元順位（大きいほど順位下がる）
            # rank_v: エージェントVの次元順位
            rank_g = None
            rank_v = None
            try:
                order_g = np.argsort(-np.asarray(content.vector, dtype=np.float64))
                rank_g = int(np.where(order_g == j_dim)[0][0]) + 1
            except Exception:
                rank_g = None
            try:
                order_v = np.argsort(-np.asarray(agent.V, dtype=np.float64))
                rank_v = int(np.where(order_v == j_dim)[0][0]) + 1
            except Exception:
                rank_v = None

            # 掘りに応じて G/V を微調整し、GPU 行列も即座に同期する
            if 0 <= j_dim < len(agent.interests):
                prev_g = agent.interests[j_dim]
                new_g = _clip01(prev_g + float(DIG_G_STEP))
                agent.interests[j_dim] = new_g
                delta_g = float(new_g - prev_g)
                if delta_g != 0.0:
                    g_updates.append((a_id, j_dim, delta_g))
            dV = random.uniform(-float(DIG_V_RANGE), float(DIG_V_RANGE))
            if 0 <= j_dim < len(agent.V):
                prev_v = agent.V[j_dim]
                new_v = _clip11(prev_v + dV)
                agent.V[j_dim] = new_v
                delta_v = float(new_v - prev_v)
                if delta_v != 0.0:
                    v_updates.append((a_id, j_dim, delta_v))

            # 形式: (step, j_dim, dV, content_g_on_j, rank_g, rank_v)
            agent.dig_log.append(
                (int(step), j_dim, float(dV), g_on_j, rank_g, rank_v)
            )

        # like 反映
        if like_flags[a_id]:
            content.likes += 1
            content.liked_by.append(a_id)
            content.like_history.append((step, a_id))
            agent.total_likes += 1

            # --- CBF用：G/I 擬似ベクトル履歴を更新 ---
            g_vec = np.asarray(content.vector,   dtype=np.float32)
            i_vec = np.asarray(content.i_vector, dtype=np.float32)
            agent.update_like_history_GI(g_vec, i_vec, step)
            # 擬似ベクトルキャッシュを無効化して次回再計算させる
            agent.pseudo_g_cache = None
            agent.pseudo_i_cache = None

            # --- CBF用：行インデックス付き履歴（CFはステージングしてキャッシュ更新時に反映） ---
            ridx = _ID2ROW.get(int(content.id)) if (_ID2ROW is not None) else None
            if ridx is not None:
                # CBF-User 用：各エージェントの like 履歴（step / content行インデックス）
                agent.lh_steps = np.append(agent.lh_steps, np.int32(step))
                agent.lh_ridx = np.append(agent.lh_ridx, np.int64(ridx))
                if agent.lh_steps.size:
                    mask = (step - agent.lh_steps) <= CF_HISTORY_WINDOW_STEPS
                    agent.lh_steps = agent.lh_steps[mask]
                    agent.lh_ridx = agent.lh_ridx[mask]
                # CF 用：キャッシュ更新時にまとめて反映するようステージング
                isc.stage_cf_like(agent.id, content.id, step)

        # dig は「確率と次元情報」をエージェント側にメモ（必要なら後で集計）
    agent.last_p_like = float(p_like_arr[a_id])
    agent.last_p_dig = float(p_dig_arr[a_id])
    agent.last_dig_dim = int(j_dims[a_id])

    if (step + 1) % 500 == 0:
        elapsed = time.time() - t0
        log_and_print(f"[GPU batch] step {step+1}/{MAX_STEPS} 経過 ({elapsed:.1f}s)", flush=True)

    if g_updates:
        a_idx = torch.tensor([t[0] for t in g_updates], dtype=torch.long, device=DEVICE)
        d_idx = torch.tensor([t[1] for t in g_updates], dtype=torch.long, device=DEVICE)
        delta = torch.tensor([t[2] for t in g_updates], dtype=torch.float32, device=DEVICE)
        engine.Ug.index_put_((a_idx, d_idx), delta, accumulate=True)
        engine.ug_matrix_t.index_put_((a_idx, d_idx), delta, accumulate=True)
    if v_updates:
        a_idx = torch.tensor([t[0] for t in v_updates], dtype=torch.long, device=DEVICE)
        d_idx = torch.tensor([t[1] for t in v_updates], dtype=torch.long, device=DEVICE)
        delta = torch.tensor([t[2] for t in v_updates], dtype=torch.float32, device=DEVICE)
        engine.Uv.index_put_((a_idx, d_idx), delta, accumulate=True)
        engine.uv_matrix_t.index_put_((a_idx, d_idx), delta, accumulate=True)

    # ============================================================================
    # ★ STEP_BINごとの多様性 / G–V相関スナップショット
    # ============================================================================
    if (step + 1) % STEP_BIN == 0:
        snap_step = step + 1

        # --- G 多様性（エントロピー / 分散）の平均 ---
        ent_list = []
        var_list = []
        gv_list  = []

        for a in agents:
            ent_list.append(_g_entropy(a.interests))
            var_list.append(_g_variance(a.interests))
            r_gv = _pearson_r(a.interests, a.V)
            if r_gv == r_gv:  # NaN でなければ
                gv_list.append(r_gv)

        avg_ent = float(sum(ent_list) / len(ent_list)) if ent_list else 0.0
        avg_var = float(sum(var_list) / len(var_list)) if var_list else 0.0

        DIVERSITY_TIMELINE.append((snap_step, avg_ent, avg_var))

        if gv_list:
            avg_gv = float(sum(gv_list) / len(gv_list))
            GV_CORR_TIMELINE.append((snap_step, avg_gv, len(gv_list)))
        else:
            GV_CORR_TIMELINE.append((snap_step, float("nan"), 0))

log_and_print("🔥 GPU バッチ版シミュレーション完了")

# ============================================================================
# 解析レポート / プロット
# ============================================================================

# --- 共通プリミティブ ---
def _print_header(title: str):
    log_line(f"=== {title} ===")

def _avg(lst):
    return (sum(lst) / len(lst)) if lst else 0.0

def _bin_index(s):
    return int(s // STEP_BIN)

def _collect_bins_range():
    return [(start, min(start + STEP_BIN, MAX_STEPS))
            for start in range(0, MAX_STEPS, STEP_BIN)]

def _vec_to_str(vec, decimals=2):
    a = np.asarray(vec, dtype=np.float32).flatten()
    fmt = f"{{:.{decimals}f}}"
    return "[" + ",".join(fmt.format(float(x)) for x in a) + "]"

def _intvec_to_str(vec):
    return "[" + ",".join(str(int(x)) for x in vec) + "]"

def _norm_or_blank(vec):
    if vec is None:
        return ""
    a = np.asarray(vec, dtype=np.float64).ravel()
    return round(float(np.linalg.norm(a)), 6)

def _vec_to_str_or_blank(vec, decimals=2):
    if vec is None:
        return ""
    return _vec_to_str(vec, decimals)

# ----[ ラベル（STEP_BINに追従） ]------------
BIN_STR = f"{STEP_BIN:,}"  # 例: 5,000 / 720 / 100

# ----[ ビルド（集計）関数群 ]----------------
# --- 集計ビルダー ---
def build_score_averages(agents, *, likes_only=True):
    """
    STEP_BIN刻みビンごとに CG/CV/I の平均スコアを算出。
    likes_only=True: いいね成立行のみ平均
               False: すべてのインプレッションで平均
    戻り値: bins, avg_cg, avg_cv, avg_ci, cnts
    """
    bins = _collect_bins_range()
    S_cg = [0.0 for _ in bins]; S_cv = [0.0 for _ in bins]; S_ci = [0.0 for _ in bins]
    Cnts = [0   for _ in bins]

    for a in agents:
        for rec in getattr(a, "score_log", []):
            s, _cid, cg, cv, ci, liked = rec
            if likes_only and (int(liked) != 1):
                continue
            bi = _bin_index(s)
            if 0 <= bi < len(bins):
                S_cg[bi] += float(cg)
                S_cv[bi] += float(cv)
                S_ci[bi] += float(ci)
                Cnts[bi] += 1

    def _safe_mean(S, N):
        out = []
        for s, n in zip(S, N):
            out.append("" if n == 0 else (s / n))
        return out

    return bins, _safe_mean(S_cg, Cnts), _safe_mean(S_cv, Cnts), _safe_mean(S_ci, Cnts), Cnts

def build_delta_dig(agents):
    """
    ΔDig（STEP_BIN刻み）
    """
    bins = _collect_bins_range()
    per_bin = [0 for _ in bins]
    for a in agents:
        for s, *_rest in getattr(a, "dig_log", []):
            bi = _bin_index(s)
            if 0 <= bi < len(per_bin):
                per_bin[bi] += 1
    cum = []
    c = 0
    for x in per_bin:
        c += x
        cum.append(c)
    return bins, per_bin, cum

def build_distance_trend(agents):
    """
    ユーザー–コンテンツ距離の推移（平均）
    impression_log: (step, cid, like_flag, cos[, pearson_r_g_content])
    cos の平均を返す（STEP_BIN刻み）
    """
    bins = _collect_bins_range()
    sums = [0.0 for _ in bins]
    cnts = [0   for _ in bins]
    for a in agents:
        for rec in getattr(a, "impression_log", []):
            # 4要素（旧）/ 5要素（新）どちらもOKにする
            if len(rec) < 4:
                continue
            s, _cid, _liked, cos = rec[:4]  # 先頭4つだけ使う
            bi = _bin_index(s)
            if 0 <= bi < len(bins):
                sums[bi] += float(cos)
                cnts[bi] += 1
    avg_cos = [(s/c if c > 0 else "") for s, c in zip(sums, cnts)]
    return bins, avg_cos, cnts

def build_user_content_corr_trend(agents):
    """
    impression_log: (step, cid, like_flag, cos, pearson_r_g_content)
    を用いて、STEP_BIN刻みで Pearson(user G, content G_active) の平均を返す。
    """
    bins = _collect_bins_range()
    sums = [0.0 for _ in bins]
    cnts = [0   for _ in bins]
    for a in agents:
        for rec in getattr(a, "impression_log", []):
            if len(rec) < 5:
                continue
            s, _cid, _liked, _cos, r = rec[:5]
            if r != r:  # NaNチェック
                continue
            bi = _bin_index(s)
            if 0 <= bi < len(bins):
                sums[bi] += float(r)
                cnts[bi] += 1
    avg_r = [(s/c if c > 0 else "") for s, c in zip(sums, cnts)]
    return bins, avg_r, cnts

def build_user_v_content_corr_trend(agents):
    """
    impression_log: (step, cid, like_flag, cos, pearson_r_g_content, pearson_v_content)
    を用いて、STEP_BIN刻みで Pearson(user V, content G_active) の平均を返す。
    """
    bins = _collect_bins_range()
    sums = [0.0 for _ in bins]
    cnts = [0   for _ in bins]
    for a in agents:
        for rec in getattr(a, "impression_log", []):
            if len(rec) < 6:
                continue
            s, _cid, _liked, _cos, _r_g, r_v = rec
            if r_v != r_v:  # NaN
                continue
            bi = _bin_index(s)
            if 0 <= bi < len(bins):
                sums[bi] += float(r_v)
                cnts[bi] += 1
    avg_r = [(s/c if c > 0 else "") for s, c in zip(sums, cnts)]
    return bins, avg_r, cnts

def build_diversity_timeline_snapshot():
    """
    G分布の多様性（分散/エントロピー）の推移
    メインループで DIVERSITY_TIMELINE に積んだ値をCSVに出せる形に
    """
    return list(DIVERSITY_TIMELINE)

def build_gv_corr_timeline_snapshot():
    """STEP_BINごとの avg Pearson(G,V) タイムラインを返す。"""
    return list(GV_CORR_TIMELINE)

def build_dig_rank_tables(agents, *, num_genres=NUM_GENRES):
    """
    掘り時の順位分布（STEP_BINごと × 1..num_genres の順位）。
    dig_log タプルの 5,6 番目が (rank_g, rank_v) として入っていれば利用。
    戻り値: bins, g_rank_table, v_rank_table, totals
      - g_rank_table[bi][r-1] = bin bi における G順位 r の件数
      - v_rank_table[bi][r-1] = bin bi における V順位 r の件数
    """
    bins = _collect_bins_range()
    K = int(num_genres)
    g_tab = [[0]*K for _ in bins]
    v_tab = [[0]*K for _ in bins]
    totals = [0 for _ in bins]

    for a in agents:
        for rec in getattr(a, "dig_log", []):
            if not rec:
                continue
            s = int(rec[0])
            bi = _bin_index(s)
            if bi < 0 or bi >= len(bins):
                continue
            # 互換対応：rank_g, rank_v が入っていなければスキップ
            if len(rec) >= 6:
                rank_g = int(rec[4]); rank_v = int(rec[5])
                if 1 <= rank_g <= K:
                    g_tab[bi][rank_g-1] += 1
                if 1 <= rank_v <= K:
                    v_tab[bi][rank_v-1] += 1
                totals[bi] += 1

    return bins, g_tab, v_tab, totals

def build_avg_dig_rank(agents, *, num_genres=NUM_GENRES):
    """
    STEP_BINごとに、掘り発生時のG順位/V順位の“平均順位”を返す。
    戻り値: bins, avg_rank_G(list or ''), avg_rank_V(list or ''), totals
    """
    bins, g_tab, v_tab, totals = build_dig_rank_tables(agents, num_genres=num_genres)
    if not bins:
        return [], [], [], []
    K = int(num_genres)
    ranks = np.arange(1, K+1, dtype=float)

    def _avg(rows, tot):
        out = []
        for row, t in zip(rows, tot):
            if t <= 0:
                out.append("")
            else:
                arr = np.array(row, dtype=float)
                out.append(float((ranks * arr).sum() / float(t)))
        return out

    return bins, _avg(g_tab, totals), _avg(v_tab, totals), totals

def _dig_counts_per_dim(agent, num_genres=NUM_GENRES):
    """
    dig_log は (step, j_dim, dV, content_g_on_j) の4タプル想定。
    旧ログが3タプルでも動くように防御実装。
    """
    cnt = [0] * num_genres
    for rec in getattr(agent, "dig_log", []):
        if not rec:
            continue
        try:
            j = int(rec[1])  # インデックス1（3/4タプル共通）
        except (IndexError, TypeError, ValueError):
            continue
        if 0 <= j < num_genres:
            cnt[j] += 1
    return cnt

def build_dig_strength_histograms(agents, *, num_bins: int = 10):
    """
    掘りが起きたときの G 次元の値（g_on_j）を 0..1 を num_bins 等分した帯域に集計。
    戻り値: bins(list), edges(np.ndarray[num_bins+1]), counts(list[list[int]])
    """
    bins = _collect_bins_range()
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    counts = [[0 for _ in range(num_bins)] for _ in bins]

    for a in agents:
        for rec in getattr(a, "dig_log", []):
            if not rec or len(rec) < 4:
                continue
            try:
                s = int(rec[0])
                g_val = float(rec[3])
            except (TypeError, ValueError):
                continue
            bi = _bin_index(s)
            if bi < 0 or bi >= len(bins):
                continue
            if not np.isfinite(g_val):
                continue
            g_clip = min(max(g_val, 0.0), 1.0)
            idx = int(np.searchsorted(edges, g_clip, side="right") - 1)
            idx = max(0, min(idx, num_bins - 1))
            counts[bi][idx] += 1

    return bins, edges, counts

# ----[ プリント（レポート）関数群 ]---------
# --- レポート出力 ---
def print_agent_summary(agents):
    n = len(agents)
    total_likes = sum(a.total_likes for a in agents)
    total_digs  = sum(len(getattr(a, "dig_log", [])) for a in agents)
    avg_likes   = total_likes / n if n else 0.0
    avg_digs    = total_digs  / n if n else 0.0

    _print_header("Agent Summary")
    log_line(f"エージェント総数        : {n}")
    log_line(f"総いいね数 (agents)    : {total_likes}")
    log_line(f"総dig数    (agents)    : {total_digs}")
    log_line(f"平均いいね数/agent     : {avg_likes:.3f}")
    log_line(f"平均dig数/agent        : {avg_digs:.3f}")

def print_avg_dig_rank_5k(agents, *, round_to_int=False):
    bins, avg_g, avg_v, totals = build_avg_dig_rank(agents)
    _print_header(f"Dig時の平均順位（{BIN_STR}刻み｜1=最上位）")
    log_line("bin_start, avg_rank_G, avg_rank_V")
    for (start, _end), g, v, _t_ignored in zip(bins, avg_g, avg_v, totals):
        g_out = "" if g == "" else f"{g:.6f}"
        v_out = "" if v == "" else f"{v:.6f}"
        log_line(f"{start}, {g_out}, {v_out}")

def print_score_averages_5k(agents, *, likes_only=True):
    label = "（いいね成立のみ）" if likes_only else "（全インプレッション）"
    bins, avg_cg, avg_cv, avg_ci, _cnts_ignored = build_score_averages(agents, likes_only=likes_only)
    _print_header(f"スコア平均 CG/CV/I {BIN_STR}刻み {label}")
    log_line("bin_start, avg_CG, avg_CV, avg_I")
    for (start, _end), g, v, i in zip(bins, avg_cg, avg_cv, avg_ci):
        g_out = "" if g == "" else f"{g:.6f}"
        v_out = "" if v == "" else f"{v:.6f}"
        i_out = "" if i == "" else f"{i:.6f}"
        log_line(f"{start}, {g_out}, {v_out}, {i_out}")

def print_delta_dig_5k(agents):
    bins, per_bin, _cum_ignored = build_delta_dig(agents)
    _print_header(f"ΔDig（{BIN_STR}刻み）")
    log_line("bin_start, delta_dig")
    for (start, _end), d in zip(bins, per_bin):
        log_line(f"{start}, {d}")

def print_distance_trend_5k(agents):
    bins, avg_cos, cnts = build_distance_trend(agents)
    _print_header(f"ユーザー–コンテンツ距離の推移（{BIN_STR}刻み）")
    log_line("bin_start, avg_cos")
    for (start, _end), cs, n in zip(bins, avg_cos, cnts):
        cs_out = ("" if cs == "" else f"{cs:.6f}")
        log_line(f"{start}, {cs_out}")

def print_diversity_timeline_summary():
    snap = build_diversity_timeline_snapshot()
    _print_header(f"G分布 多様性の推移（{BIN_STR}刻みスナップショット）")
    log_line("step, avg_entropy, avg_variance")
    for step, h, v in snap:
        log_line(f"{step}, {h:.6f}, {v:.6f}")

def print_user_content_corr_5k(agents):
    bins, avg_r, cnts = build_user_content_corr_trend(agents)
    _print_header(f"ユーザーG–コンテンツG（active）Pearson相関の推移（{BIN_STR}刻み）")
    log_line("bin_start, avg_pearson_r")
    for (start, _end), r, n in zip(bins, avg_r, cnts):
        r_out = ("" if r == "" else f"{float(r):.6f}")
        log_line(f"{start}, {r_out}")

def print_user_v_content_corr_5k(agents):
    bins, avg_r, cnts = build_user_v_content_corr_trend(agents)
    _print_header(f"ユーザーV–コンテンツG（active）Pearson相関の推移（{BIN_STR}刻み）")
    log_line("bin_start, avg_pearson_r_v")
    for (start, _end), r, n in zip(bins, avg_r, cnts):
        r_out = ("" if r == "" else f"{float(r):.6f}")
        log_line(f"{start}, {r_out}")

def print_gv_corr_timeline_summary():
    snap = build_gv_corr_timeline_snapshot()
    _print_header(f"G–V 相関タイムライン（{BIN_STR}刻みスナップショット）")
    log_line("step, avg_pearson_gv")
    for step, avg_r, _n_ignored in snap:
        r_out = "" if (avg_r != avg_r) else f"{avg_r:.6f}"
        log_line(f"{step}, {r_out}")

def print_dig_strength_hist(agents, *, num_bins: int = 10):
    bins, edges, counts = build_dig_strength_histograms(agents, num_bins=num_bins)
    _print_header(f"Dig回数（G強度帯域別, {BIN_STR}刻み）")
    band_labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(num_bins)]
    log_line("bin_start, " + ", ".join(band_labels))
    for (start, _end), row in zip(bins, counts):
        log_line(f"{start}, " + ", ".join(str(int(x)) for x in row))

# ----[ エクスポート関数群 ]-----------------
def build_panel_10x20(agents):
    """
    10人 × 各時間帯(start..start+19) × 20 impressions の content_id（不足 -1）
    """
    bins = _collect_bins_range()
    step2cid = {aid:{} for aid in PANEL_AGENT_IDS}
    for aid in PANEL_AGENT_IDS:
        a = agents[aid]
        for s, cid, *_rest in getattr(a, "impression_log", []):
            step2cid[aid][int(s)] = int(cid)
    rows = []  # (bin_start, agent_id, [cid_20])
    for (start, _end) in bins:
        for aid in PANEL_AGENT_IDS:
            arr = [step2cid[aid].get(t, -1) for t in range(start, min(start+20, MAX_STEPS))]
            while len(arr) < 20:
                arr.append(-1)
            rows.append((start, aid, arr))
    return rows

def export_panel_csv(agents, contents, prefix=None):
    """
    各STEP_BINごとに、先頭10人(PANEL_AGENT_IDS)×連続20ステップの表示コンテンツを
    1行でまとめてCSV出力。各スロットに content_id と G/I ベクトル文字列を併記。

    出力列: bin_start, agent_id, (cid_0, G_0, I_0), ..., (cid_19, G_19, I_19)
    - G_k / I_k は `_vec_to_str(vec, 2)` の文字列（例: "[0.12,0.34,...]")
    - 欠損（cid=-1）は空文字 "" を出力
    """
    # 1) ルックアップ辞書（id -> (G, I)）
    lut = {int(c.id): (c.vector, c.i_vector) for c in contents}

    # 2) データ行（既存の10×20の骨格を流用）
    rows = build_panel_10x20(agents)

    # 3) 出力
    prefix = prefix or OUT_PREFIX
    path = f"{prefix}_panel_10agents_20impressions_per_{STEP_BIN}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        # ヘッダ: bin_start, agent_id, [cid_k, G_k, I_k]×20
        header = ["bin_start", "agent_id"]
        for k in range(20):
            header += [f"cid_{k}", f"G_{k}", f"I_{k}"]
        w.writerow(header)

        # 本文
        for (start, aid, arr) in rows:
            row_out = [start, aid]
            for cid in arr:
                if cid >= 0 and cid in lut:
                    g_vec, i_vec = lut[cid]
                    row_out.append(int(cid))
                    row_out.append(_vec_to_str(g_vec, 2))  # "[...]" 形式
                    row_out.append(_vec_to_str(i_vec, 2))
                else:
                    # 欠損スロット（-1など）
                    row_out += ["", "", ""]
            w.writerow(row_out)

    log_and_print(f"✅ パネルCSV（G/I付き）を書き出しました -> {path}")

def export_readable_csv(agents, contents, prefix=None):
    """
    agents.csv / contents.csv を読みやすい列設計で出力（ヘッダと列の完全一致版）
    - like_flag_log など未定義属性は使用しない
    - contents.csv の列順も論理順に修正
    """
    final_step = MAX_STEPS - 1
    # レポートはコサイン固定
    REPORT_SIM_LABEL = "cos (alpha=1)"

    # --- agents.csv ---
    prefix = prefix or OUT_PREFIX
    with open(f"{prefix}_agents.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        sim_label_g = f"cosine(pseudo_G, G_final) [{REPORT_SIM_LABEL}]"
        sim_label_i = f"cosine(pseudo_I, I_final) [{REPORT_SIM_LABEL}]"

        # ヘッダ（列順と出力の完全一致）
        w.writerow([
            "agent_id","total_likes",
            "digs_total","digs_per_dim",
            "norm_G_init","norm_G_final",
            "norm_I_init","norm_I_final",
            "norm_V_init","norm_V_final",
            "G_entropy_init","G_entropy_final",
            "G_var_init","G_var_final",
            "initial_G","final_G",
            "initial_V","final_V",
            "initial_I","final_I",
            "norm_pseudo_G_final","norm_pseudo_I_final",
            "pseudo_G_final","pseudo_I_final",
            sim_label_g, sim_label_i ,
            "pearson(G_final, V_final)"
        ])

        for a in agents:
            # dig 分布ベクトル
            dig_cnt_vec = _dig_counts_per_dim(a, NUM_GENRES)
            digs_total  = int(sum(dig_cnt_vec))

            # 擬似ベクトル（最終時点）
            try:
                pg = a._compute_pseudo(a.like_history_G, final_step)
            except AttributeError:
                pg = a.compute_pseudo_vector_G(final_step)
            try:
                pi = a._compute_pseudo(a.like_history_I, final_step)
            except AttributeError:
                pi = a.compute_pseudo_vector_I(final_step)

            # コサイン固定で出力
            sim_pg_g = _safe_sim_alpha(pg, a.interests, 1.0)
            sim_pi_i = _safe_sim_alpha(pi, a.I,         1.0)
            out_pg = "" if sim_pg_g == "" else round(float(sim_pg_g), 6)
            out_pi = "" if sim_pi_i == "" else round(float(sim_pi_i), 6)

            # 多様性（初期/最終）
            gH_init  = _g_entropy(a.initial_vector)
            gH_final = _g_entropy(a.interests)
            gV_init  = _g_variance(a.initial_vector)
            gV_final = _g_variance(a.interests)

            # ここで Pearson(G_final, V_final) を計算（NaNは空文字に）
            gv_corr_final = _pearson_r(a.interests, a.V)
            if gv_corr_final != gv_corr_final or gv_corr_final is None:  # NaNチェック
                gv_out = ""
            else:
                gv_out = round(float(gv_corr_final), 6)

            w.writerow([
                a.id, int(a.total_likes),
                digs_total, _intvec_to_str(dig_cnt_vec),
                _norm_or_blank(a.initial_vector), _norm_or_blank(a.interests),
                _norm_or_blank(a.initial_I),      _norm_or_blank(a.I),
                _norm_or_blank(a.initial_V),      _norm_or_blank(a.V),
                round(gH_init, 6), round(gH_final, 6),
                round(gV_init, 6), round(gV_final, 6),
                _vec_to_str(a.initial_vector, 2), _vec_to_str(a.interests, 2),
                _vec_to_str(a.initial_V, 2),      _vec_to_str(a.V, 2),
                _vec_to_str(a.initial_I, 2),      _vec_to_str(a.I, 2),
                _norm_or_blank(pg), _norm_or_blank(pi),
                _vec_to_str_or_blank(pg, 2), _vec_to_str_or_blank(pi, 2),
                out_pg, out_pi,
                gv_out
            ])

    # --- contents.csv ---
    with open(f"{prefix}_contents.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        # ヘッダを実データ順に合わせる
        w.writerow([
            "content_id","views","likes",
            "content_G","content_I",
            "trend_score","buzz_score",
            "norm_content_G","norm_content_I"
        ])
        for c in contents:
            w.writerow([
                int(c.id), int(c.views), int(c.likes),
                _vec_to_str(c.vector, 2), _vec_to_str(c.i_vector, 2),
                round(c.trend_score, 6),
                round(c.get_buzz_score(MAX_STEPS-1), 6),
                _norm_or_blank(c.vector), _norm_or_blank(c.i_vector),
            ])

    log_and_print(f"✅ CSVを書き出しました -> {prefix}_agents.csv, {prefix}_contents.csv")

# ============================================================================
# 図出力：各タイムライン（STEP_BIN刻み）
# ============================================================================

# --- プロット ---
def _to_nan_array(vals):
    """'' を np.nan に置換してプロット可能に"""
    arr = []
    for v in vals:
        if v == "" or v is None:
            arr.append(np.nan)
        else:
            try:
                arr.append(float(v))
            except Exception:
                arr.append(np.nan)
    return np.array(arr, dtype=float)

def plot_score_avgs_by_time(agents, *, likes_only=True, out_path=None):
    """
    スコア平均（CG, CV, I）を同一図に3本の折れ線で。
    x軸=bin開始ステップ, y軸=平均スコア
    """
    bins, avg_cg, avg_cv, avg_ci, cnts = build_score_averages(agents, likes_only=likes_only)
    x = np.array([start for (start, _end) in bins], dtype=float)
    y_cg = _to_nan_array(avg_cg)
    y_cv = _to_nan_array(avg_cv)
    y_ci = _to_nan_array(avg_ci)

    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    plt.plot(x, y_cg, marker="o", label="avg_CG")
    plt.plot(x, y_cv, marker="o", label="avg_CV")
    plt.plot(x, y_ci, marker="o", label="avg_I")
    plt.xlabel(f"time (bin={BIN_STR} steps)")
    plt.ylabel("average score")
    tail = "likes only" if likes_only else "all impressions"
    plt.title(f"Score averages over time ({tail})")
    plt.legend(frameon=True)
    plt.tight_layout()
    if out_path is None:
        suffix = "likes" if likes_only else "all"
        out_path = f"{OUT_PREFIX}_score_avgs_by_time_{suffix}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_path}")
    return out_path

# 置き換え：ΔDigのみを描画（累積なし）
def plot_delta_dig_by_time(agents, *, out_path=None):
    bins, per_bin, _cum_ignored = build_delta_dig(agents)
    x = np.array([start for (start, _end) in bins], dtype=float)
    y_delta = np.array(per_bin, dtype=float)

    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    plt.plot(x, y_delta, marker="o", label="ΔDig per bin")
    plt.xlabel(f"time (bin={BIN_STR} steps)")
    plt.ylabel("count")
    plt.title("ΔDig over time")
    plt.legend(frameon=True)
    plt.tight_layout()
    if out_path is None:
        out_path = f"{OUT_PREFIX}_delta_dig_by_time.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_path}")
    return out_path

def plot_avg_user_content_cos_by_time(agents, *, out_path=None):
    """
    ユーザー–コンテンツ距離（平均コサイン類似）1本線。
    impression_log 由来の cos を bin 平均。
    """
    bins, avg_cos, cnts = build_distance_trend(agents)
    x = np.array([start for (start, _end) in bins], dtype=float)
    y = _to_nan_array(avg_cos)

    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    plt.plot(x, y, marker="o", label="avg cos(user G, content G)")
    plt.xlabel(f"time (bin={BIN_STR} steps)")
    plt.ylabel("average cosine similarity")
    plt.title("User–Content average cosine similarity over time")
    plt.legend(frameon=True)
    plt.tight_layout()
    if out_path is None:
        out_path = f"{OUT_PREFIX}_avg_user_content_cos_by_time.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_path}")
    return out_path

def plot_avg_user_content_corr_by_time(agents, *, out_path=None):
    """
    ユーザーG–コンテンツG（active）Pearson相関の平均を1本線で描画。
    """
    bins, avg_r, cnts = build_user_content_corr_trend(agents)
    x = np.array([start for (start, _end) in bins], dtype=float)
    y = []
    for v in avg_r:
        if v == "" or v is None:
            y.append(np.nan)
        else:
            try:
                y.append(float(v))
            except Exception:
                y.append(np.nan)
    y = np.array(y, dtype=float)

    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    plt.plot(x, y, marker="o", label="avg Pearson r (user G vs content G_active)")
    plt.xlabel(f"time (bin={BIN_STR} steps)")
    plt.ylabel("average Pearson r")
    plt.title("User–Content Pearson correlation over time")
    plt.legend(frameon=True)
    plt.tight_layout()
    if out_path is None:
        out_path = f"{OUT_PREFIX}_avg_user_content_pearson_by_time.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_path}")
    return out_path

def plot_avg_user_v_content_corr_by_time(agents, *, out_path=None):
    """
    ユーザーV–コンテンツG（active）Pearson相関の平均を1本線で描画。
    """
    bins, avg_r, cnts = build_user_v_content_corr_trend(agents)
    x = np.array([start for (start, _end) in bins], dtype=float)
    y = _to_nan_array(avg_r)

    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    plt.plot(x, y, marker="o", label="avg Pearson r (user V vs content G_active)")
    plt.xlabel(f"time (bin={BIN_STR} steps)")
    plt.ylabel("average Pearson r")
    plt.title("User V – Content G Pearson correlation over time")
    plt.legend(frameon=True)
    plt.tight_layout()
    if out_path is None:
        out_path = f"{OUT_PREFIX}_avg_user_v_content_pearson_by_time.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_path}")
    return out_path

def plot_g_diversity_timelines(*, out_entropy=None, out_variance=None):
    """
    G多様性タイムライン：
      - エントロピーと分散は“別図”で出力（要求通り二つの図）。
    """
    snap = build_diversity_timeline_snapshot()
    if not snap:
        log_and_print("⚠️ DIVERSITY_TIMELINE が空です（メインループのSTEP_BINスナップショットを確認）。")
        return None, None

    steps = np.array([s for (s, _h, _v) in snap], dtype=float)
    ent   = np.array([h for (_s, h, _v) in snap], dtype=float)
    var   = np.array([v for (_s, _h, v) in snap], dtype=float)

    # エントロピー
    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    plt.plot(steps, ent, marker="o", label="entropy (normalized)")
    plt.xlabel(f"time (snapshot every {BIN_STR} steps)")
    plt.ylabel("entropy (0..1)")
    plt.title("G diversity timeline – Entropy")
    plt.legend(frameon=True)
    plt.tight_layout()
    if out_entropy is None:
        out_entropy = f"{OUT_PREFIX}_G_entropy_timeline.png"
    plt.savefig(out_entropy, dpi=160)
    plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_entropy}")

    # 分散
    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    plt.plot(steps, var, marker="o", label="variance")
    plt.xlabel(f"time (snapshot every {BIN_STR} steps)")
    plt.ylabel("variance")
    plt.title("G diversity timeline – Variance")
    plt.legend(frameon=True)
    plt.tight_layout()
    if out_variance is None:
        out_variance = f"{OUT_PREFIX}_G_variance_timeline.png"
    plt.savefig(out_variance, dpi=160)
    plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_variance}")

    return out_entropy, out_variance

def plot_gv_corr_timeline(*, out_path=None):
    snap = build_gv_corr_timeline_snapshot()
    if not snap:
        log_and_print("⚠️ GV_CORR_TIMELINE が空です。")
        return None
    x = np.array([s for (s, _r, _n) in snap], dtype=float)
    y = np.array([np.nan if (r != r) else float(r) for (_s, r, _n) in snap], dtype=float)

    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    plt.plot(x, y, marker="o", label="avg Pearson r (G vs V)")
    plt.xlabel(f"time (snapshot every {BIN_STR} steps)")
    plt.ylabel("average Pearson r")
    plt.title("Agent-level G–V correlation over time")
    plt.legend(frameon=True)
    plt.tight_layout()
    if out_path is None:
        out_path = f"{OUT_PREFIX}_avg_gv_corr_timeline.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_path}")
    return out_path


def plot_avg_dig_rank_trend(agents, *, out_path=None):
    """
    STEP_BINごとの“平均順位”の推移（G平均・V平均の2本線）。
    値は小さいほど上位に掘っていることを意味する。
    """
    bins, g_tab, v_tab, totals = build_dig_rank_tables(agents)
    if not bins:
        log_and_print("⚠️ dig順位データがありません。")
        return None
    K = len(g_tab[0])

    x = np.array([start for (start, _end) in bins], dtype=float)
    def _avg_rank(rows, tot):
        out = []
        for row, t in zip(rows, tot):
            if t <= 0:
                out.append(np.nan)
            else:
                # 加重平均： sum(rank * count) / total
                ranks = np.arange(1, K+1, dtype=float)
                out.append(float(np.dot(ranks, np.array(row, dtype=float))) / float(t))
        return np.array(out, dtype=float)

    y_g = _avg_rank(g_tab, totals)
    y_v = _avg_rank(v_tab, totals)

    plt.figure(figsize=(9, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    plt.plot(x, y_g, marker="o", label="avg rank (G)")
    plt.plot(x, y_v, marker="o", label="avg rank (V)")
    plt.xlabel(f"time (bin={BIN_STR} steps)")
    plt.ylabel("average rank (lower is higher)")
    plt.title("Average dig rank over time")
    plt.legend(frameon=True)
    plt.tight_layout()
    if out_path is None:
        out_path = f"{OUT_PREFIX}_avg_dig_rank_trend.png"
    plt.savefig(out_path, dpi=160); plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_path}")
    return out_path

def plot_dig_strength_hist_by_time(agents, *, num_bins: int = 10, out_path=None):
    """
    STEP_BIN ごとに、G強度帯域（0..1を等分）の掘り回数を折れ線で重ねて描画。
    色は古い順→新しい順にグラデーション。
    """
    bins, edges, counts = build_dig_strength_histograms(agents, num_bins=num_bins)
    if not bins:
        log_and_print("⚠️ digデータがありません。")
        return None

    x = (edges[:-1] + edges[1:]) / 2.0  # 帯域中心（点の数 = num_bins）
    colors = plt.cm.plasma(np.linspace(0, 1, len(bins)))

    plt.figure(figsize=(10, 6))
    for idx, ((start, _end), row) in enumerate(zip(bins, counts)):
        y = np.asarray(row, dtype=float)
        plt.plot(x, y, marker="o", linestyle="-", color=colors[idx], label=f"{start}-{_end-1}")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(edges, [f"{e:.1f}" for e in edges])
    plt.xlabel("G strength band (0-1)")
    plt.ylabel("dig count")
    plt.title("Dig count per G strength band (per time bin)")

    # カラーバーで時間のグラデーションを示す
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=bins[0][0], vmax=bins[-1][0]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
    cbar.set_label("bin start step")

    # legend は多くなるので省略（カラーバーで代替）
    plt.tight_layout()
    if out_path is None:
        out_path = f"{OUT_PREFIX}_dig_strength_hist_by_time.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    log_and_print(f"✅ 図を書き出しました -> {out_path}")
    return out_path

def plot_avg_dig_rank(agents, *, out_path=None):
    """
    STEP_BIN ごとに集計された平均順位（GとV）を同じグラフに描画する。
    y軸は「平均順位（1 = 最も強い次元）」。
    """
    bins, avg_g, avg_v, totals = build_avg_dig_rank(agents)
    if not bins:
        log_and_print("⚠️ digデータがありません。")
        return None

    # x軸（stepの開始位置）
    x = [start for (start, _end) in bins]

    # y軸（平均順位）— 値が '' のところは NaN にする
    import numpy as np
    y_g = np.array([np.nan if g == "" else float(g) for g in avg_g], dtype=float)
    y_v = np.array([np.nan if v == "" else float(v) for v in avg_v], dtype=float)

    # プロット
    plt.figure(figsize=(9, 5))
    plt.plot(x, y_g, marker="o", linewidth=2, label="avg rank (G)", color="tab:blue")
    plt.plot(x, y_v, marker="o", linewidth=2, label="avg rank (V)", color="tab:orange")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel(f"time (per {STEP_BIN} steps)")
    plt.ylabel("average rank (1 = strongest)")
    plt.title("Average Dig Rank Over Time (G & V)")
    plt.legend()
    plt.tight_layout()

    if out_path is None:
        out_path = f"{OUT_PREFIX}_avg_dig_rank.png"

    plt.savefig(out_path, dpi=160)
    plt.close()
    log_and_print(f"✅ 平均順位グラフを書き出しました -> {out_path}")
    return out_path


def emit_all_outputs(agents, contents, *, out_prefix=OUT_PREFIX):
    """
    すべての出力関数を一括で実行:
      - コンソール出力（print_*）
      - CSV出力（panel + agents/contents）
      - 図出力（plot_*）
    """

    # 1) コンソール（プリント）—— “関数があるものは全部”
    print_agent_summary(agents)
    print_delta_dig_5k(agents)
    print_distance_trend_5k(agents)
    print_diversity_timeline_summary()
    print_score_averages_5k(agents, likes_only=True)
    print_score_averages_5k(agents, likes_only=False)
    print_user_content_corr_5k(agents)          # ユーザーG–コンテンツG(active)の相関タイムライン
    print_user_v_content_corr_5k(agents)        # ユーザーV–コンテンツG(active)の相関タイムライン
    print_gv_corr_timeline_summary()            # G–V 相関タイムライン（avg Pearson r、n_agents）
    print_avg_dig_rank_5k(agents, round_to_int=True)
    print_dig_strength_hist(agents, num_bins=10)

    # 2) CSVエクスポート
    export_panel_csv(agents, contents=contents, prefix=out_prefix)
    export_readable_csv(agents, contents=contents, prefix=out_prefix)

    # 3) 図出力（plot_* 全部）
    try:
        plot_score_avgs_by_time(agents, likes_only=True,  out_path=f"{out_prefix}_score_avgs_by_time_likes.png")
        plot_score_avgs_by_time(agents, likes_only=False, out_path=f"{out_prefix}_score_avgs_by_time_all.png")
        plot_delta_dig_by_time(agents, out_path=f"{out_prefix}_delta_dig_by_time.png")
        plot_avg_user_content_cos_by_time(agents, out_path=f"{out_prefix}_avg_user_content_cos_by_time.png")
        plot_avg_user_content_corr_by_time(agents, out_path=f"{out_prefix}_avg_user_content_pearson_by_time.png")
        plot_avg_user_v_content_corr_by_time(agents, out_path=f"{out_prefix}_avg_user_v_content_pearson_by_time.png")
        plot_gv_corr_timeline(out_path=f"{out_prefix}_avg_gv_corr_timeline.png")
        plot_g_diversity_timelines(
            out_entropy=f"{out_prefix}_G_entropy_timeline.png",
            out_variance=f"{out_prefix}_G_variance_timeline.png"
        )
        plot_avg_dig_rank_trend(agents, out_path=f"{out_prefix}_avg_dig_rank_trend.png")
        plot_dig_strength_hist_by_time(agents, num_bins=10, out_path=f"{out_prefix}_dig_strength_hist_by_time.png")
    except Exception as e:
        log_and_print(f"⚠️ 図の書き出しに失敗しました: {e}")

# シミュレーション完了後の集約出力
emit_all_outputs(agents, isc.pool, out_prefix=OUT_PREFIX)
log_and_print(f"✅ ログを書き出しました -> {LOG_FILE_PATH}")
write_log_file(LOG_FILE_PATH)
