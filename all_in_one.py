
import numpy as np
from collections import namedtuple
# 1. DINH NGHIA HANH DONG VA CAC BAN DO
# 4 hanh dong di chuyen: 0: Len, 1: Xuong, 2: Trai, 3: Phai
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Len", "Xuong", "Trai", "Phai"]
State = namedtuple("State", ["row", "col", "progress", "switch_on", "skip_used"])
state = State

MAPS = {
    "level1": {
        "ten": "Level 1 - De (5x5)",
        "grid": [
            "#S...",
            "#.A..",
            "#..B.",
            "C....",
            "...X."
        ],
        "chain": [["A"], ["B"], ["C"]],  
        "locked": [],
    },
    "level2": {
        "ten": "Level 2 - Trung binh (7x7, co re nhanh)",
        "grid": [
            "#S....#",
            "#.A...#",
            "#..B..#",
            "#.....#",
            "#..C..#",
            "#.....#",
            "#...D.X"
        ],
        "chain": [["A"], ["B", "C"], ["D"]], 
        "locked": [],
    },
    "level3": {
        "ten": "Level 3 - Kho (8x8, co cong tac W va o di tat K)",
        "grid": [
            "########",
            "#S.....#",
            "#.A.K..#",
            "#..B...#",
            "#....W.#",
            "#....C.#",
            "#......#",
            "#D...X.#"
        ],
        "chain": [["A"], ["B"], ["C"], ["D"]],
        "locked": ["C"], 
    },
}

# 2. HAM XAY DUNG MOI TRUONG (DANG TU DIEN - KHONG DUNG CLASS)

def tao_moi_truong(level_name="level1"):
    """
    Khoi tao moi truong bang mot Dictionary don gian.
    Chua thong tin ban do, vi tri cac vat pham va danh sach trang thai.
    """
    spec = MAPS[level_name]
    grid = spec["grid"]
    so_dong = len(grid)
    so_cot = len(grid[0])

    # Tim vi tri cac o dac biet tren ban do
    start = None
    exit_pos = None
    switch_pos = None
    skip_pos = None
    items = {}  # Luu vi tri cac chu cai kho bau: (dong, cot) -> 'A'

    for r in range(so_dong):
        for c in range(so_cot):
            ky_tu = grid[r][c]
            if ky_tu == "S":
                start = (r, c)
            elif ky_tu == "X":
                exit_pos = (r, c)
            elif ky_tu == "W":
                switch_pos = (r, c)
            elif ky_tu == "K":
                skip_pos = (r, c)
            elif ky_tu in ["A", "B", "C", "D"]:
                items[(r, c)] = ky_tu

    # Dong goi thong tin moi truong vao 1 dictionary
    env = {
        "level_name": level_name,
        "ten": spec["ten"],
        "grid": grid,
        "so_dong": so_dong,
        "so_cot": so_cot,
        "chain": spec["chain"],
        "locked": spec["locked"],
        "start": start,
        "exit": exit_pos,
        "switch": switch_pos,
        "skip": skip_pos,
        "items": items,
        "step_cost": -1.0,  # Chi phi cho moi buoc di
    }

    # Tim tat ca cac trang thai ma Agent co the dat toi (Reachable States)
    trang_thai_dau = State(start[0], start[1], 0, 0, 0)
    danh_sach_states = [trang_thai_dau]
    state_to_idx = {trang_thai_dau: 0}

    hang_doi = [trang_thai_dau]
    while len(hang_doi) > 0:
        hien_tai = hang_doi.pop(0)
        for a in range(4):
            tiep_theo, _, _ = buoc_di(env, hien_tai, a)
            if tiep_theo not in state_to_idx:
                state_to_idx[tiep_theo] = len(danh_sach_states)
                danh_sach_states.append(tiep_theo)
                hang_doi.append(tiep_theo)

    env["states"] = danh_sach_states
    env["state_to_idx"] = state_to_idx
    env["n_states"] = len(danh_sach_states)
    env["n_actions"] = 4
    return env


def buoc_di(env, state, action):
#dinh nghia state va lap toa do state trong moi truong
    r, c, progress, sw_on, k_used = state
    dr, dc = ACTIONS[action]
    nr = r + dr
    nc = c + dc
    
    # 1. Kiem tra dam vao tuong hoac di ra ngoai mep ban do
    if nr < 0 or nr >= env["so_dong"] or nc < 0 or nc >= env["so_cot"] or env["grid"][nr][nc] == "#":
        # Dung yen tai cho, bi phat nhe do dam tuong
        return state, -3.0, False

    moi_sw = sw_on
    moi_k = k_used
    moi_progress = progress

    # 2. Dam vao o cong tac W (mo khoa kho bau o Level 3)
    if (nr, nc) == env["switch"] and moi_sw == 0:
        moi_sw = 1
    # 3. Dam vao o di tat K (duoc phep bo qua 1 kho bau o Level 3)
    if (nr, nc) == env["skip"] and moi_k == 0 and moi_progress < len(env["chain"]):
        moi_k = 1
        moi_progress += 1

    # 4. Dam vao o co kho bau ('A', 'B', 'C', 'D')
    if (nr, nc) in env["items"]:
        item = env["items"][(nr, nc)]

        # Truong hop 4.1: Kho bau dang bi khoa (chua bat cong tac W)
        if item in env["locked"] and moi_sw == 0:
            return state, -20.0, False  # Phạt và không cho đi vào

        # Truong hop 4.2: Nhat dung kho bau can tim o giai doan hien tai
        if moi_progress < len(env["chain"]) and item in env["chain"][moi_progress]:
            moi_progress += 1
            trang_thai_moi = State(nr, nc, moi_progress, moi_sw, moi_k)
            return trang_thai_moi, 10.0, False  # Thuong +10 diem vi nhat dung

        # Truong hop 4.3: Nhat sai kho bau (nhat vuot thu tu truoc)
        cac_item_sau = []
        for giai_doan in env["chain"][moi_progress + 1:]:
            for it in giai_doan:
                cac_item_sau.append(it)
        if moi_progress < len(env["chain"]) and item in cac_item_sau:
            return state, -20.0, False  # Phat nang vi nhat sai thu tu

    # 5. Dam vao o cua thoat 'X'
    if (nr, nc) == env["exit"]:
        # Neu da nhat du tat ca kho bau theo yeu cau
        if moi_progress >= len(env["chain"]):
            trang_thai_moi = State(nr, nc, moi_progress, moi_sw, moi_k)
            return trang_thai_moi, 100.0, True  # Thang game! Thuong lon +100 diem
        else:
            # Chua nhat du ma da doi ve dich -> phat vi thoat som
            return state, -20.0, False

    # 6. Buoc di hop le vao o trong binh thuong
    trang_thai_moi = State(nr, nc, moi_progress, moi_sw, moi_k)
    return trang_thai_moi, env["step_cost"], False

    
def tao_mo_hinh_P(env):
    """
    Tao ma tran chuyen trang thai P cho cac thuat toan quy hoach dong (VI va PI).
    P[s][a] = [(xac_suat=1.0, trang_thai_ke, phan_thuong, da_xong)]
    """
    P = {}
    for s_idx, state in enumerate(env["states"]):
        P[s_idx] = {}
        for a in range(4):
            next_state, reward, done = buoc_di(env, state, a)
            next_idx = env["state_to_idx"][next_state]
            P[s_idx][a] = [(1.0, next_idx, reward, done)]
    return P

# =====================================================================
# 3. 5 THUAT TOAN HOC TANG CUONG (RL ALGORITHMS)
# =====================================================================

# -------------------------------------------------------------
# Thuat toan 1: VALUE ITERATION (Quy hoach dong - Model-based)
# -------------------------------------------------------------
def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Lap gia tri: lien tuc cap nhat gia tri V(s) theo phuong trinh toi uu Bellman:
    V(s) = max_a sum(P * (R + gamma * V(s')))
    """
    P = tao_mo_hinh_P(env)
    n_states = env["n_states"]
    V = np.zeros(n_states)

    while True:
        delta = 0.0
        for s in range(n_states):
            # Tinh gia tri cho tung hanh dong a
            gia_tri_cac_a = []
            for a in range(4):
                gia_tri_a = 0.0
                for prob, next_s, reward, done in P[s][a]:
                    if done:
                        gia_tri_a += prob * reward
                    else:
                        gia_tri_a += prob * (reward + gamma * V[next_s])
                gia_tri_cac_a.append(gia_tri_a)

            # Chon gia tri lon nhat
            v_tot_nhat = max(gia_tri_cac_a)
            delta = max(delta, abs(V[s] - v_tot_nhat))
            V[s] = v_tot_nhat

        # Khi do thay doi nho hon nguong theta thi dung
        if delta < theta:
            break

    # Rut trich chinh sach toi uu: policy[s] = hanh dong mang lai gia tri cao nhat
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        gia_tri_cac_a = []
        for a in range(4):
            gia_tri_a = 0.0
            for prob, next_s, reward, done in P[s][a]:
                if done:
                    gia_tri_a += prob * reward
                else:
                    gia_tri_a += prob * (reward + gamma * V[next_s])
            gia_tri_cac_a.append(gia_tri_a)
        policy[s] = np.argmax(gia_tri_cac_a)

    return policy


# -------------------------------------------------------------
# Thuat toan 2: POLICY ITERATION (Quy hoach dong - Model-based)
# -------------------------------------------------------------
def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    Lap chinh sach: gom 2 buoc luan phien:
    1. Policy Evaluation: tinh gia tri V(s) theo chinh sach hien tai.
    2. Policy Improvement: cai thien chinh sach bang cach chon hanh dong tot nhat.
    """
    P = tao_mo_hinh_P(env)
    n_states = env["n_states"]
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)  # Khoi tao chinh sach ban dau

    while True:
        # Buoc 1: Danh gia chinh sach (Policy Evaluation)
        while True:
            delta = 0.0
            for s in range(n_states):
                a = policy[s]
                v_moi = 0.0
                for prob, next_s, reward, done in P[s][a]:
                    if done:
                        v_moi += prob * reward
                    else:
                        v_moi += prob * (reward + gamma * V[next_s])
                delta = max(delta, abs(V[s] - v_moi))
                V[s] = v_moi
            if delta < theta:
                break

        # Buoc 2: Cai thien chinh sach (Policy Improvement)
        chinh_sach_on_dinh = True
        for s in range(n_states):
            hanh_dong_cu = policy[s]

            # Tim hanh dong tot nhat dua tren V vua tinh duoc
            gia_tri_cac_a = []
            for a in range(4):
                val = 0.0
                for prob, next_s, reward, done in P[s][a]:
                    if done:
                        val += prob * reward
                    else:
                        val += prob * (reward + gamma * V[next_s])
                gia_tri_cac_a.append(val)

            hanh_dong_moi = np.argmax(gia_tri_cac_a)
            if hanh_dong_cu != hanh_dong_moi:
                chinh_sach_on_dinh = False
            policy[s] = hanh_dong_moi

        # Neu chinh sach khong con thay doi nua thi dung
        if chinh_sach_on_dinh:
            break

    return policy


# -------------------------------------------------------------
# Thuat toan 3: Q-LEARNING (Model-free, Off-policy TD Control)
# -------------------------------------------------------------
def q_learning(env, episodes=10000, gamma=0.99, alpha=0.1, eps_start=1.0, eps_end=0.05, seed=42):
    """
    Hoc Q-table truc tiep tu tuong tac ma khong can mo hinh P.
    Cong thuc cap nhat:
    Q(s, a) = Q(s, a) + alpha * [Reward + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    rng = np.random.default_rng(seed)
    n_states = env["n_states"]
    Q = np.zeros((n_states, 4))
    start_state = env["states"][0]
    start_idx = 0

    for ep in range(episodes):
        s_tuple = start_state
        s_idx = start_idx

        # Giam dan ti le kham pha epsilon
        epsilon = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / (episodes * 0.8))

        done = False
        step_count = 0
        while not done and step_count < 100:
            step_count += 1

            # Epsilon-greedy: chon ngau nhien hoac chon hanh dong tot nhat hien tai
            if rng.random() < epsilon:
                action = rng.integers(4)
            else:
                action = np.argmax(Q[s_idx])

            next_tuple, reward, done = buoc_di(env, s_tuple, action)
            next_idx = env["state_to_idx"][next_tuple]

            # Cong thuc Q-learning (Off-policy dung max_a')
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_idx])

            Q[s_idx, action] += alpha * (target - Q[s_idx, action])

            s_tuple = next_tuple
            s_idx = next_idx

    # Chinh sach la hanh dong co Q-value lon nhat tai moi trang thai
    return np.argmax(Q, axis=1)


# -------------------------------------------------------------
# Thuat toan 4: SARSA (Model-free, On-policy TD Control)
# -------------------------------------------------------------
def sarsa(env, episodes=10000, gamma=0.99, alpha=0.1, eps_start=1.0, eps_end=0.05, seed=42):
    """
    Tuong tu Q-learning nhung on-policy:
    Q(s, a) cap nhat theo hanh dong thuc su duoc chon tiep theo a' tu epsilon-greedy:
    Q(s, a) = Q(s, a) + alpha * [Reward + gamma * Q(s', a') - Q(s, a)]
    """
    rng = np.random.default_rng(seed)
    n_states = env["n_states"]
    Q = np.zeros((n_states, 4))
    start_state = env["states"][0]
    start_idx = 0

    for ep in range(episodes):
        s_tuple = start_state
        s_idx = start_idx

        epsilon = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / (episodes * 0.8))

        # Chon hanh dong ban dau a
        if rng.random() < epsilon:
            action = rng.integers(4)
        else:
            action = np.argmax(Q[s_idx])

        done = False
        step_count = 0
        while not done and step_count < 100:
            step_count += 1
            next_tuple, reward, done = buoc_di(env, s_tuple, action)
            next_idx = env["state_to_idx"][next_tuple]

            # Chon hanh dong tiep theo next_action (a')
            if rng.random() < epsilon:
                next_action = rng.integers(4)
            else:
                next_action = np.argmax(Q[next_idx])

            # Cong thuc SARSA (On-policy dung Q(next_idx, next_action))
            if done:
                target = reward
            else:
                target = reward + gamma * Q[next_idx, next_action]

            Q[s_idx, action] += alpha * (target - Q[s_idx, action])

            s_tuple = next_tuple
            s_idx = next_idx
            action = next_action

    return np.argmax(Q, axis=1)


# -------------------------------------------------------------
# Thuat toan 5: MONTE CARLO (Model-free, First-Visit Control)
# -------------------------------------------------------------
def monte_carlo(env, episodes=10000, gamma=0.99, alpha=0.1, eps_start=1.0, eps_end=0.05, seed=42):
    """
    Monte Carlo First-Visit: khong dung uoc luong buoc tiep theo (khong bootstrap),
    ma chay het 1 tap de thu thap toan bo chuoi hanh dong, roi cap nhat nguoc ve:
    G_t = R_{t+1} + gamma * G_{t+1}
    Q(s_t, a_t) = Q(s_t, a_t) + alpha * [G_t - Q(s_t, a_t)]
    """
    rng = np.random.default_rng(seed)
    n_states = env["n_states"]
    Q = np.zeros((n_states, 4))
    start_state = env["states"][0]
    start_idx = 0

    for ep in range(episodes):
        epsilon = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / (episodes * 0.8))

        # 1. Chay het 1 episode va luu lai quy dao: [(s_idx, action, reward), ...]
        quy_dao = []
        s_tuple = start_state
        s_idx = start_idx
        done = False
        step_count = 0

        while not done and step_count < 100:
            step_count += 1
            if rng.random() < epsilon:
                action = rng.integers(4)
            else:
                action = np.argmax(Q[s_idx])

            next_tuple, reward, done = buoc_di(env, s_tuple, action)
            next_idx = env["state_to_idx"][next_tuple]

            quy_dao.append((s_idx, action, reward))
            s_tuple = next_tuple
            s_idx = next_idx

        # 2. Duyet nguoc tu cuoi tap ve dau de cap nhat Q-value
        G = 0.0
        da_xet = set()
        for s_t, a_t, r_t in reversed(quy_dao):
            G = r_t + gamma * G
            # First-visit: chi cap nhat cho lan xuat hien dau tien cua cap (s, a)
            if (s_t, a_t) not in da_xet:
                da_xet.add((s_t, a_t))
                Q[s_t, a_t] += alpha * (G - Q[s_t, a_t])

    return np.argmax(Q, axis=1)

# =====================================================================
# 4. DANH GIA CHINH SACH VA IN KET QUA
# =====================================================================

def chay_thu_nghiem(env, policy):
   
    s_tuple = env["states"][0]
    s_idx = 0
    tong_thuong = 0.0
    so_buoc = 0
    done = False
    duong_di = [(s_tuple[0], s_tuple[1])]

    while not done and so_buoc < 100:
        a = policy[s_idx]
        next_tuple, reward, done = buoc_di(env, s_tuple, a)
        next_idx = env["state_to_idx"][next_tuple]

        tong_thuong += reward
        so_buoc += 1
        s_tuple = next_tuple
        s_idx = next_idx
        duong_di.append((s_tuple[0], s_tuple[1]))
    state_final = s_tuple
    return tong_thuong, so_buoc, done, duong_di,state_final
    


def main():
    print("=" * 70)
    print("DE TAI 08: SAN TIM KHO BAU THEO THU TU (ORDERED TREASURE HUNT)")
    print("5 THUAT TOAN TABULAR RL: VI, PI, Q-LEARNING, SARSA, MONTE CARLO")
    print("=" * 70)

    levels = ["level1", "level2", "level3"]
    danh_sach_thuat_toan = [
        ("Value Iteration", lambda env: value_iteration(env)),
        ("Policy Iteration", lambda env: policy_iteration(env)),
        ("Q-Learning", lambda env: q_learning(env, episodes=10000)),
        ("SARSA", lambda env: sarsa(env, episodes=10000)),
        ("Monte Carlo", lambda env: monte_carlo(env, episodes=10000)),
    ]

    for lvl in levels:
        env = tao_moi_truong(lvl)
        print(f"\n[+] {env['ten'].upper()} (|S| = {env['n_states']} trang thai, Luoi: {env['so_dong']}x{env['so_cot']})")
        print(f"{'Thuat toan':<20} | {'Diem (Return)':<14} | {'So buoc':<8} | {'Thanh cong':<10}")
        print("-" * 62)

        for ten_tt, ham_tt in danh_sach_thuat_toan:
            chinh_sach = ham_tt(env)
            diem, so_buoc, thanh_cong, _, state_final = chay_thu_nghiem(env, chinh_sach)
            print(f"{ten_tt:<20} | {diem:<14.1f} | {so_buoc:<8} | {str(thanh_cong):<10}")
        print("Trang thai ket thuc:", state_final)
    # Minh hoa quy dao toi uu tren Level 3 (Kho)
    print("\n" + "=" * 70)
    print("MINH HOA DUONG DI TOI UU CUA LEVEL 3 (Policy Iteration)")
    print("Thu tu dung: A -> Bat cong tac W -> B -> C -> D -> Vao cua X")
    print("=" * 70)
    env3 = tao_moi_truong("level3")
    pi3 = policy_iteration(env3)
    diem3, buoc3, ok3, path3, state_final = chay_thu_nghiem(env3, pi3)
    print(f"Ket qua: Thanh cong = {ok3} trong {buoc3} buoc (Diem tong cong = {diem3:.1f})")
    print("Cac toa do di qua (dong, cot):")
    print(" -> ".join(f"({r},{c})" for r, c in path3))
    

if __name__ == "__main__":
    main()
