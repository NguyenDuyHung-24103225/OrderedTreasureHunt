# AGENTS.md — Đề tài 08: Ordered Treasure Hunt (RL, cuối kỳ)

## What this repo is
Single-student RL coursework project. The assignment is `true (10).pdf` (31 topics); **only topic 08 is in scope**.
Sibling repos `../th3` and `../NguyenDuyHung/RL_24103225_NguyenDuyHung` are earlier labs by the same
student and define the house style — read their `README.md` / `src/*.py` before inventing a new convention.

## Reading the assignment (the agent cannot read the PDF directly)
The `Read` tool fails on PDFs for this model. Extract text instead — topic 08 is PDF **page 8**:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import pdfplumber;print('\n'.join((p.extract_text() or '') for p in pdfplumber.open('true (10).pdf').pages[7:8]))"
```

### Spec — ĐỀ TÀI 08 (condensed from PDF p.8)
- Goal: collect items in a valid order, then escape.
- MDP: `State = (row, col, task_progress, flags)`; `Action = up / down / left / right` (4 actions).
- Reward: bonus for taking the **correct next** item, penalty for touching an **out-of-order** item, large bonus on completion.
- Required settings (the project uses **3 square maps** that cover all four cases):
  1. `level1` (5×5): chain `A→B→C`.
  2. `level2` (7×7): two valid chains `A→B→D` and `A→C→D`.
  3. `level3` (8×8): chain `A→B→C→D` where `C` is locked behind a switch, plus a bonus tile that skips exactly 1 chain element.
- **Topic-specific proof required**: the same cell with a different task progress is a *different state* (demonstrate it).
- PDF's own algorithm list is "Policy Iteration and Q-learning"; the project scope is **5 algorithms** (below).
- Mandatory experiment requirements (from the PDF, all apply): ≥3 configs of increasing difficulty; report `|S|` and `|A|`;
  spell out transition, terminal condition and the extra state variables that preserve the Markov property; run many seeds/episodes;
  plot learning curves or return statistics; visualise the final policy on ≥2 settings; compare ≥2 algorithms and analyse
  failure / slow-convergence cases; try ≥2 reward or env-parameter configurations and explain how the policy changes;
  ship env + agent source, map/config files, a report, and a demo of one episode driven by the learned policy.
- **Hard ban: no DQN, no function approximation → tabular only.**

## The 5 algorithms (scope)
`Value Iteration`, `Policy Iteration`, `Q-learning` (TD control, off-policy), `SARSA` (TD control, on-policy),
`Monte Carlo` (on-policy first-visit control).

All five must share one uniform interface so `main.py` can sweep and compare them.

VI and PI are **model-based**: they need `P(s'|s,a)` and `R`. A custom `gymnasium.Env` has **no** `.P` —
`env.unwrapped.P` exists only for gymnasium's built-in classic-control/toy-text envs (sibling `../th3/Lab02/src/mdp_utils.py`
relies on it for FrozenLake; it does **not** apply here). The env must expose its own model, e.g.
`env.build_transition_model() -> P[s][a] = [(prob, s_next, reward, terminal), ...]`, and VI/PI consume that.

## Repo layout (evidence-based — do not churn)
A prior session left stale bytecode in `__pycache__/` at the repo root (folder since deleted, names recovered):
`envs.py`, `agents.py`, `configs.py` — and `configs.py`'s `co_names` were `('__doc__', 'CONFIGS', 'get_config')`.
That is the intended module split. The project is now built on that split:

```
cuoiky/
  true (10).pdf          # assignment, read-only
  AGENTS.md  README.md  requirements.txt
  envs.py       # OrderedTreasureHuntEnv(gymnasium.Env), level/map definitions, transition model
  agents.py     # the 5 algorithms behind one uniform fit() API
  configs.py    # CONFIGS dict + get_config(name) + REWARD_PROFILES
  plots.py      # every figure of the report (env, policies, curves, animation)
  main.py       # CLI entry point: train / compare / render / autoplay
  build_notebook.py          # generates notebooks/*.ipynb from the .py sources
  make_report_docx.py        # generates reports/*.docx (per-line code explanation)
  make_report_pptx.py        # generates reports/*.pptx (numbers read from results.json)
  notebooks/    # OrderedTreasureHunt_24103225_NguyenDuyHung.ipynb
  figures/      # generated *.png and *.gif
  reports/      # generated *.docx, *.pptx
  results.json  # the numbers of the experiment, read by both report generators
```

Sibling repos nest python under `src/` and name notebooks `LabNN_<MSSV>_<Name>.ipynb`; the root-level placement
above is what this repo's own bytecode indicates.

**Single source of truth.** The `.py` files are the source. `build_notebook.py` copies their source verbatim into
the notebook, so *never* edit the notebook by hand and *never* duplicate code inside it — edit the `.py` and re-run
`python build_notebook.py`. Every number in the Word and PowerPoint reports comes from `results.json`, which
`python main.py --compare` writes; never type a result by hand.

## House style (from the sibling repos — match it)
- **Code comments and docstrings: ASCII only, no Vietnamese diacritics** (`mc_utils.py` uses ASCII transliteration,
  `mdp_utils.py` is English). **User-facing content is the opposite: it must be full Vietnamese with diacritics** —
  level titles and descriptions, figure titles and axis labels, notebook prose, and all the prose of the Word and
  PowerPoint reports. A report full of "moi truong" / "thuoc" is a defect; a comment full of diacritics is a defect too.
- Any script that prints Vietnamese to the console must first call `sys.stdout.reconfigure(encoding="utf-8")`
  (see the cp1252 gotcha below) — `main.py` and the two report generators already do it.
- Section banners: a run of `# ====...` lines with an ALL-CAPS title between them (see `mdp_utils.py`).
- Reproducibility: explicit `seed` argument, `np.random.default_rng(seed)` (not the legacy global `np.random.seed`).
- VI/PI convergence knobs named `gamma`, `theta=1e-8`, `max_iterations`; VI also returns the per-iteration `deltas`.
- Arrow glyphs live in a dict named `ACTION_SYMBOLS` and are used when printing a policy grid.
- Student identity (from sibling READMEs): Nguyễn Duy Hưng, MSSV 24103225, lớp EEE.AI, GitHub `NguyenDuyHung-24103225`.

## Toolchain (verified on this machine)
- Interpreter: system **Python 3.14.0** at `C:\Users\Acer\AppData\Local\Python\bin\python.exe`. No venv in this repo.
- `pip` is **not** on PATH → always `python -m pip install ...`. The package index is reachable.
- Preinstalled: `gymnasium` 1.3.0, `numpy` 2.3.4, `matplotlib` 3.11.1, `pandas`, `seaborn`, `pygame-ce` 2.5.8
  (import name is `pygame`), `python-docx` 1.2.0, `python-pptx` 1.0.2, `pdfplumber`, `pymupdf`, `nbconvert` 7.17.1,
  `nbclient`, `ipykernel`.
- **Not** installed: `jupyterlab`, `jupyter_server`, `notebook`, `pytest`, `ruff`, `black`, `imageio`.
  → `python -m pip install jupyterlab` if an interactive notebook server is needed.
- Import `gymnasium`, never the legacy `gym` (0.26.2 is also installed and is easy to import by mistake).
- Headless notebook run that works today:
  `python -m jupyter nbconvert --to notebook --execute --inplace notebooks\<name>.ipynb`
  It rewrites the notebook with outputs. Expect `MissingIDFieldWarning` unless every cell carries an `id` field.
- Precedent for generating notebooks: sibling `../th3/build_notebook.py` writes the `.ipynb` JSON programmatically.

### Gotchas that will bite you
- The Windows console is **cp1252**; printing Vietnamese raises `UnicodeEncodeError`. Fix per script with
  `sys.stdout.reconfigure(encoding="utf-8")`, or set `$env:PYTHONIOENCODING="utf-8"` for the whole shell.
- matplotlib renders Vietnamese diacritics correctly with the default DejaVu Sans, so figure labels may be accented.
- matplotlib's animation writers here are **pillow + html only** → `anim.save("x.mp4")` fails. GIF works out of the box
  via `PillowWriter`. To get mp4: copy
  `...\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe` into a folder **renamed to `ffmpeg.exe`** and
  prepend that folder to `PATH` (the bundled filename is not `ffmpeg.exe`, so PATH alone is not enough);
  `matplotlib.animation.writers.list()` then reports `['pillow','ffmpeg','ffmpeg_file','html']`.
- pygame's default font (`freesansbold.ttf`) has **no Vietnamese glyphs** → load `C:\Windows\Fonts\arial.ttf` for any
  Vietnamese label in the auto-play window.

## Required deliverables (beyond the PDF)
1. **Notebook** `notebooks/*.ipynb` holding the real implementation and experiments (not a summary). Generate it the way
   `../th3/build_notebook.py` does rather than hand-editing notebook JSON.
2. **Word report** `reports/*.docx` explaining **every line of code**. Produce it with a script that reads the source
   files and emits per-line explanations, so it can be regenerated after any edit — do not hand-type it, or it will
   drift out of sync.
3. **PowerPoint** `reports/*.pptx` for the presentation. Not deep, but it **must** contain: the MDP of this problem,
   the environment description, the action set, and **numeric results**. Every number in the deck must be read from the
   results file that `main.py` writes — never hand-typed.
4. **Graphics**: render the environment (grid, agent, items, policy arrows) and provide a **self-play / auto-play mode**
  that steps the learned policy through the environment. This is also the PDF's minimum deliverable
  "chương trình demo một episode bằng policy đã học".

## CLI contract (keep stable — the docx/pptx generators call it)
```
python main.py --algo {vi,pi,qlearning,sarsa,mc} --config <name> --episodes N --seed S
python main.py --compare --out results.json
python main.py --render --config <name> --algo <name> [--out figures/x.png]
python main.py --autoplay --algo <name> --config <name> [--gif out.gif]
```
`--compare` must write machine-readable JSON (mean/std return, success rate, steps-to-goal, wall-clock) so the report
generators can fill their tables without re-running anything. `main.run_comparison()` is the shared entry point used
by both the CLI and the notebook, so the two can never disagree.

Config names are `level1` (5×5, chain A→B→C), `level2` (7×7, two valid chains A→B→D and A→C→D),
`level3` (8×8, chain A→B→C→D with a locked item and a bonus tile). Reward profiles are `default`,
`cheap_steps` (step penalty −0.2) and `expensive_steps` (step penalty −5.0). The reward experiment and the
seed experiment both run on `level3`.

## Definition of done
- `python main.py --compare` runs all 5 algorithms over the 3 configs and writes `results.json`
  (including the reward experiment and the seed experiment).
- The notebook executes top-to-bottom headlessly and produces every figure in `figures/`.
- `reports/*.docx` is regenerated from the current sources (no stale line numbers).
- `reports/*.pptx` numbers match `results.json` exactly.
- Auto-play renders a learned-policy episode on ≥2 configs.

## Bài báo IEEE conference (`report/ieee_conference/`)
Bản bài báo theo template IEEE nằm riêng trong `report/ieee_conference/`:

```
main.tex                    # nguồn LaTeX, biên dịch bằng XeLaTeX hoặc LuaLaTeX
IEEEtran.cls                # lớp của template IEEE conference
make_ieee_report.py         # sinh bản Word cùng nội dung, số liệu đọc từ results.json
figures/                    # ảnh dùng trong bài, copy từ figures/ của project
IEEE_Conference_OrderedTreasureHunt.docx
```

- **`main.tex` phải biên dịch bằng XeLaTeX hoặc LuaLaTeX** (file dùng `fontspec` với `TeX Gyre Termes`).
  Trên Overleaf: đổi *Compiler* sang XeLaTeX. `pdflatex` mặc định **không** chạy được vì file dùng `fontspec`.
- **Bài viết bằng tiếng Anh đơn giản, tầm trình độ B1–B2** — cùng giọng với bản nháp gốc. Không dùng từ học thuật khó
  ("degenerate", "equilibrium analysis", "trajectory", "asymptotic", "ablation", "formulation", "leverage"...). Câu ngắn
  (khoảng 14–16 từ), luôn kèm số liệu cụ thể. `python check_english.py` đo độ dài câu và liệt kê từ khó để kiểm tra.
- Mọi con số trong bài lấy từ `results.json`; mọi ảnh lấy từ `figures/` của project. Không gõ tay, không vẽ lại hình.
- Không có LaTeX cài trên máy này, nên PDF phải biên dịch trên Overleaf rồi tải về đặt cạnh file `.tex`.
