# Đề tài 08 - Ordered Treasure Hunt

## Thông tin sinh viên

- Họ tên: Nguyễn Duy Hưng
- MSSV: 24103225
- Lớp: EEE.AI - Học tăng cường (Reinforcement Learning)

## Bài toán

Agent phải thu thập các vật phẩm theo một thứ tự hợp lệ rồi mới thoát ra qua ô `X`.
Chạm vào vật phẩm sai thứ tự sẽ bị phạt, lấy đúng vật phẩm tiếp theo sẽ được thưởng.
Môi trường là một lưới ASCII nhỏ, viết bằng `gymnasium.Env` (không dùng DQN, không dùng function approximation).

## Mô hình MDP

| Thành phần | Mô tả |
|---|---|
| State | `(row, col, task_progress, flags)` - vị trí agent, số phần tử của chuỗi đã giải, bitmask của các biến phụ và của các vật phẩm đã giải |
| Action | `0 = up`, `1 = down`, `2 = left`, `3 = right` (4 hành động) |
| Reward | `+10` lấy đúng vật phẩm tiếp theo, `-20` chạm vật phẩm sai thứ tự, `-8` vật phẩm bị khóa, `-20` vào ô thoát sớm, `-3` đụng tường, `-1` mỗi bước, `+100` khi thoát |
| Terminal | Agent bước vào ô `X` khi `task_progress` đã đạt giá trị lớn nhất |
| Truncation | Episode kết thúc sau `max_steps = 100` bước |

**Vì sao `task_progress` phải thuộc về state?** Ở level 1, xét ô `(2, 1)`:

- với `task_progress = 0` thì vật phẩm `A` còn trên lưới, bước vào ô đó cho `+10` và tiến độ tăng lên `1`;
- với `task_progress = 1` thì vật phẩm `A` đã bị lấy, ô đó đã thành sàn, bước vào chỉ cho `-1`.

Cùng một ô nhưng hai lần chuyển trạng thái khác nhau, do đó bắt buộc phải là hai state khác nhau.
Các biến phụ `switch_on` (level 3), `skip_used` (level 4) và mask của các vật phẩm đã giải (level 2) được thêm vào state với lý do tương tự.

## Ba setting của đề tài

| Setting | Nội dung | `\|S\|` | `\|A\|` |
|---|---|---|---|
| `level1` | Một chuỗi `A -> B -> C`, lưới 5×5 | 79 | 4 |
| `level2` | Hai chuỗi hợp lệ khác nhau: `A -> B -> D` và `A -> C -> D`, lưới 7×7 | 199 | 4 |
| `level3` | Chuỗi `A -> B -> C -> D`, vật phẩm `C` bị khóa sau công tác `W`, ô thưởng `K` bỏ qua được 1 phần tử của chuỗi, lưới 8×8 | 693 | 4 |

Cả ba lưới đều là hình vuông. `|S|` là số trạng thái **có thể đạt được**, được tính bằng tìm cây theo bề rộng từ trạng thái bắt đầu.

## Năm thuật toán

| Thuật toán | Loại | Chính sách | Ghi chú |
|---|---|---|---|
| Value Iteration | Dựa trên mô hình | Off policy | Dùng toán tử Bellman tối ưu, hội tụ đúng vài vòng lặp |
| Policy Iteration | Dựa trên mô hình | Off policy | Luân phiên đánh giá chính sách và cải thiện chính sách |
| Q-learning | Không mô hình | Off policy | Bootstrap bằng `max_a Q(s',a)` |
| SARSA | Không mô hình | On policy | Bootstrap bằng `Q(s',a')` của chính sách hành vi |
| Monte Carlo | Không mô hình | On policy | First visit, không bootstrap, cần episode hoàn chỉnh |

Tất cả cùng một giao diện `fit(env, **params)`, nên `main.py` so sánh được chúng bằng một đường code duy nhất.

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

## Cách chạy

```bash
# Huấn luyện một thuật toán trên một setting
python main.py --algo qlearning --config level1 --episodes 20000 --seed 42

# In ra policy tối ưu dạng lưới mũi tên
python main.py --algo pi --config level3 --policy

# So sánh 5 thuật toán trên 3 setting, ghi kết quả và vẽ hình
python main.py --compare --out results.json

# Vẽ môi trường kèm policy đã học
python main.py --render --config level3 --algo pi --out figures/policy_level3_pi.png

# Chế độ tự chơi: chạy một episode bằng policy đã học, lưu thành GIF
python main.py --autoplay --algo qlearning --config level1 --gif figures/autoplay_level1.gif
```

## Sinh lại notebook và các báo cáo

```bash
python build_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\OrderedTreasureHunt_24103225_NguyenDuyHung.ipynb

python make_report_docx.py   # reports\Giai_thich_ma_nguon.docx
python make_report_pptx.py   # reports\Bai_thuyet_trinh.pptx
```

- Notebook chứa **toàn bộ mã nguồn** của project (được `build_notebook.py` chèn từ các file `.py`) và chạy lại toàn bộ thí nghiệm.
- Báo cáo Word giải thích từng dòng mã, được sinh tự động từ mã nguồn nên không bao giờ lệch với code.
- File PowerPoint lấy mọi con số từ `results.json` nên luôn khớp với thí nghiệm đã chạy.

## Kết quả

Mean return của policy tham lam, đánh giá trên 200 episode, seed 42, 20000 episode huấn luyện:

| Thuật toán | level1 | level2 | level3 |
|---|---|---|---|
| Value Iteration | 122.00 | 123.00 | 128.00 |
| Policy Iteration | 122.00 | 123.00 | 128.00 |
| Q-learning | 122.00 | 123.00 | 120.00 |
| SARSA | 122.00 | 123.00 | 120.00 |
| Monte Carlo | 122.00 | 123.00 | 119.00 |

Tỷ lệ thành công là `100%` trên cả ba setting với mọi thuật toán.

## Nhận xét

- Môi trường có tính chuyển trạng thái **xác định**, nên Value Iteration và Policy Iteration tìm ra policy tối ưu chính xác, còn ba thuật toán không mô hình chỉ tiệm cận.
- Trên level 1 và level 2 cả năm thuật toán đều đạt kết quả tối ưu, vì bài toán chỉ cần tránh một vài ô bị phạt.
- **Level 3 mới phân biệt được các thuật toán.** Chuỗi `A -> B -> C -> D` dài hơn, vật phẩm `C` bị khóa sau công tác `W`, và ô thưởng `K` nằm ngay cạnh vật phẩm `A` nên rất dễ bước vào.
  - Value Iteration và Policy Iteration tìm ra đường lấy đủ bốn vật phẩm, return `128.00`.
  - SARSA và Monte Carlo rút ngắn qua ô thưởng, bỏ mất vật phẩm `B`, chỉ được `118.00` tới `120.00`.
  - Monte Carlo vướng vào một cực điểm cục bộ: cập nhật bằng trung bình dần nên nó giữ nguyên đường rút ngắn và không thử lại đo vòng. Hình `figures/local_optimum_level3.png` so sánh hai policy.
- **Kết quả còn phụ thuộc vào seed.** Thí nghiệm trên level 3 với ba seed (42, 7, 123) cho thấy Value Iteration và Policy Iteration luôn đạt `128.00`, còn Q-learning thoát ra khỏi cực điểm cục bộ ở hai seed trong ba seed, còn SARSA và Monte Carlo không seed nào thoát ra. Bảng và hình: `figures/seed_experiment_level3.png`.
- **Ảnh hưởng của reward.** Chỉ thay chi phí của một bước trên level 3, policy tối ưu đã đổi hoàn toàn:
  - `cheap_steps` (chi phí `-0.2`): bỏ qua một vật phẩm không đáng, agent đi vòng để lấy đủ bốn vật phẩm, return `137.60`.
  - `expensive_steps` (chi phí `-5.0`): bỏ qua một vật phẩm là tốt hơn, agent dùng ô thưởng `K`, return `80.00`.
  Hình `figures/reward_profiles_level3.png` cho thấy mũi tên ở ô cạnh `K` đổi hướng.
- Hình ảnh nằm trong thư mục `figures/`: môi trường và policy của ba setting, đường học, đường hội tụ của hai thuật toán dùng mô hình, biểu đồ so sánh, hình chứng minh tính Markov, kết quả nhiều seed và file GIF của chế độ tự chơi.

## Cấu trúc thư mục

```
cuoiky/
  true (10).pdf       # đề bài, để nguyên
  AGENTS.md           # hướng dẫn cho các phiên làm việc khác
  README.md
  requirements.txt
  envs.py             # môi trường OrderedTreasureHuntEnv và 3 setting
  agents.py           # 5 thuật toán sau cùng một giao diện fit()
  configs.py          # cấu hình thí nghiệm và các hồ sơ reward
  plots.py            # toàn bộ hình vẽ của báo cáo
  main.py             # giao diện dòng lệnh
  build_notebook.py   # sinh notebook từ mã nguồn
  make_report_docx.py # sinh báo cáo Word giải thích từng dòng mã
  make_report_pptx.py # sinh file PowerPoint trình bày
  notebooks/          # notebook đầy đủ mã nguồn và thí nghiệm
  figures/            # các hình *.png và *.gif
  reports/            # các file *.docx và *.pptx
  results.json        # kết quả số của thí nghiệm
```

## Tài liệu tham khảo

- Sutton & Barto, *Reinforcement Learning: An Introduction* (chương 3, 6, 7).
- Tài liệu môn học Học tăng cường - đề tài 08 trong file `true (10).pdf`.
