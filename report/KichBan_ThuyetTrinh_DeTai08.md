# KỊCH BẢN THUYẾT TRÌNH BÁO CÁO ĐỒ ÁN HỌC TĂNG CƯỜNG (REINFORCEMENT LEARNING)
## ĐỀ TÀI 08: ORDERED TREASURE HUNT (THU THẬP KHO BÁU THEO THỨ TỰ)
### BỘ SLIDE TINH GỌN (14 SLIDES TRỌNG TÂM)

---

### **THÔNG TIN CHUNG**
- **Học phần:** Học Tăng Cường (Reinforcement Learning)
- **Đề tài:** Đề tài số 08 - *Ordered Treasure Hunt*
- **Sinh viên báo cáo:** Nguyễn Duy Hưng - MSV: 24103225 - Lớp: EEE.AI
- **Giảng viên hướng dẫn:** TS. Vũ Hoàng Diệu
- **Thời lượng thuyết trình:** ~10 - 15 phút (trung bình 45 - 60 giây / slide)
- **File trình chiếu:** `report/ThuyetTrinh_DeTai08_OrderedTreasureHunt.pptx` (14 slide chuẩn 16:9)

---

## MỤC LỤC KỊCH BẢN 14 SLIDE

1. [Slide 1: Trang tiêu đề & Giới thiệu đồ án](#slide-1-trang-tiêu-đề--giới-thiệu-đồ-án)
2. [Slide 2: Tổng quan bài toán & Ràng buộc thứ tự](#slide-2-tổng-quan-bài-toán--ràng-buộc-thứ-tự)
3. [Slide 3: Mô hình toán học MDP của bài toán](#slide-3-mô-hình-toán-học-mdp-của-bài-toán)
4. [Slide 4: Không gian trạng thái 5 chiều (State Space)](#slide-4-không-gian-trạng-thái-5-chiều-state-space)
5. [Slide 5: Cách tính chất Markov được đảm bảo trong đề tài](#slide-5-cách-tính-chất-markov-được-đảm-bảo-trong-đề-tài)
6. [Slide 6: Không gian hành động & Quy luật chuyển dịch](#slide-6-không-gian-hành-động--quy-luật-chuyển-dịch)
7. [Slide 7: Thiết kế hàm phần thưởng (Reward Engineering)](#slide-7-thiết-kế-hàm-phần-thưởng-reward-engineering)
8. [Slide 8: Ba cấu hình bản đồ thử nghiệm (C1, C2, C3)](#slide-8-ba-cấu-hình-bản-đồ-thử-nghiệm-c1-c2-c3)
9. [Slide 9: Thuật toán 1: Policy Iteration (Model-based DP)](#slide-9-thuật-toán-1-policy-iteration-model-based-dp)
10. [Slide 10: Thuật toán 2: Q-Learning (Model-free Off-policy TD)](#slide-10-thuật-toán-2-q-learning-model-free-off-policy-td)
11. [Slide 11: Thuật toán 3: SARSA (Model-free On-policy TD)](#slide-11-thuật-toán-3-sarsa-model-free-on-policy-td)
12. [Slide 12: Kết quả thực nghiệm đối chuẩn hiệu năng](#slide-12-kết-quả-thực-nghiệm-đối-chuẩn-hiệu-năng)
13. [Slide 13: Trực quan hóa quỹ đạo tối ưu trên bản đồ 2D](#slide-13-trực-quan-hóa-quỹ-đạo-tối-ưu-trên-bản-đồ-2d)
14. [Slide 14: Tổng kết đồ án và bài học kinh nghiệm](#slide-14-tổng-kết-đồ-án-và-bài-học-kinh-nghiệm)

---

### Slide 1: Trang tiêu đề & Giới thiệu đồ án
- **Thời lượng:** ~30 giây
- **Lời thoại:**
  > *"Kính thưa quý Thầy Cô trong Hội đồng và các bạn sinh viên. Hôm nay, em xin phép được trình bày báo cáo đồ án kết thúc học phần môn Học Tăng Cường (Reinforcement Learning) với Đề tài số 08 mang tên: **Ordered Treasure Hunt** – Bài toán thu thập kho báu theo thứ tự ràng buộc. Đồ án được hướng dẫn bởi TS. Vũ Hoàng Diệu. Sau đây em xin phép bắt đầu phần trình bày."*

---

### Slide 2: Tổng quan bài toán & Ràng buộc thứ tự
- **Thời lượng:** ~50 giây
- **Lời thoại:**
  > *"Điểm cốt lõi khác biệt của Ordered Treasure Hunt so với bài toán tìm đường thông thường nằm ở **ràng buộc thứ tự cứng (Hard Sequence Constraint)**. Tác tử xuất phát từ ô Start, phải nhặt lần lượt từng kho báu theo đúng chuỗi quy định (ví dụ $A \rightarrow B \rightarrow C$). Nếu tác tử bước vào kho báu sai thứ tự, hoặc chạm vào vật phẩm đang bị khóa, tác tử sẽ bị xử phạt ngay lập tức. Đặc biệt, nếu tác tử bước vào Cổng Exit khi chưa nhặt đủ đồ, tập chơi sẽ kết thúc trong thất bại và bị trừ phạt nặng. Toàn bộ bài toán được giải quyết bằng phương pháp **Tabular Reinforcement Learning**, đảm bảo tính hội tụ toán học tuyệt đối mà không cần dùng mạng nơ-ron xấp xỉ hàm."*

---

### Slide 3: Mô hình toán học MDP của bài toán
- **Thời lượng:** ~45 giây
- **Lời thoại:**
  > *"Về mặt lý thuyết, bài toán được mô hình hóa hoàn chỉnh dưới dạng Quá trình quyết định Markov hữu hạn (Finite MDP) thông qua bộ ngũ $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$. Hệ thống có tính chất chuyển dịch tất định 100% với hệ số chiết khấu tương lai $\gamma = 0.99$. Vòng lặp episode dừng lại khi tác tử bước vào ô Exit (dù thành công hay thất bại) hoặc khi chạm ngưỡng tối đa 200 bước."*

---

### Slide 4: Không gian trạng thái 5 chiều (State Space)
- **Thời lượng:** ~60 giây
- **Lời thoại:**
  > *"Để hệ thống thỏa mãn tính chất Markov, không gian trạng thái được thiết kế gồm 5 chiều rời rạc: $s = (r, c, p, sw, tok)$. Trong đó: $r$ và $c$ là tọa độ hàng và cột; $p$ là chỉ số tiến độ nhiệm vụ – đại diện cho tập tiền tố các kho báu đã nhặt đúng; $sw$ là trạng thái bật/tắt của công tắc mở khóa; và $tok$ là trạng thái 3 mức của viên ngọc Skip-Token (chưa nhặt, đang giữ, hoặc đã dùng). Quy mô không gian trạng thái trên 3 bản đồ thực nghiệm lần lượt là: C1 gồm 100 states; C2 gồm 343 states; và C3 với các cơ chế công tắc/token mở rộng gồm 1.920 states."*

---

### Slide 5: Cách tính chất Markov được đảm bảo trong đề tài
- **Thời lượng:** ~60 giây
- **Lời thoại:**
  > *"Đây là đóng góp toán học then chốt của đề tài: Nếu chúng ta chỉ dùng tọa độ 2D $(r, c)$, bài toán sẽ bị vi phạm tính chất Markov nghiêm trọng. Lý do là khi tác tử đứng tại ô kho báu $B$, việc nhặt $B$ là 'hợp lệ được thưởng' hay 'sai luật bị phạt' hoàn toàn phụ thuộc vào việc trong quá khứ tác tử đã nhặt $A$ hay chưa. Để khắc phục, nhóm áp dụng kỹ thuật **Mở rộng trạng thái (State Augmentation)**: Đóng gói toàn bộ lịch sử cần thiết vào 3 biến bổ sung $p, sw, tok$. Nhờ đó, tại bất kỳ thời điểm nào, tác tử chỉ cần nhìn vào trạng thái hiện tại là có đầy đủ thông tin để ra quyết định tối ưu mà không cần nhớ quá khứ."*

---

### Slide 6: Không gian hành động & Quy luật chuyển dịch
- **Thời lượng:** ~40 giây
- **Lời thoại:**
  > *"Không gian hành động gồm 4 hướng di chuyển rời rạc: Lên (0), Xuống (1), Trái (2), Phải (3). Quy luật chuyển dịch là tất định: Nếu tác tử bước vào đường đi thông thoáng thì sẽ sang ô kế tiếp với xác suất 1.0; nếu đâm vào tường đá hoặc mép biên, tác tử sẽ bị cản lại đứng yên tại chỗ và bị trừ điểm phạt va chạm."*

---

### Slide 7: Thiết kế hàm phần thưởng (Reward Engineering)
- **Thời lượng:** ~60 giây
- **Lời thoại:**
  > *"Hàm phần thưởng được thiết kế nhằm định hình hành vi cho tác tử: Phạt nhẹ mỗi bước đi (-0.05) để ép tìm đường ngắn nhất; phạt đâm tường (-0.10); thưởng nhặt đúng kho báu (+1.0) nhằm dẫn dắt tác tử đi đúng chuỗi; phạt nhặt sai (-1.0); thưởng gạt công tắc và nhặt token (+0.5); và đại thưởng +10.0 khi hoàn thành xuất sắc tại Exit. Điểm đặc biệt nhất là mức **phạt thoát non -5.0**: Nếu không có mức phạt này, tác tử sẽ học được hành vi tiêu cực là chạy thẳng ra cửa Exit để kết thúc sớm nhằm trốn tránh mức phạt bước đi. Mức phạt -5.0 đã triệt tiêu hoàn toàn bẫy bỏ cuộc sớm này."*

---

### Slide 8: Ba cấu hình bản đồ thử nghiệm (C1, C2, C3)
- **Thời lượng:** ~45 giây
- **Lời thoại:**
  > *"Nhóm thử nghiệm trên 3 bản đồ có độ phức tạp tăng dần: C1 là bản đồ 5x5 cơ bản với chuỗi đơn $A \rightarrow B \rightarrow C$; C2 là bản đồ 7x7 mở rộng cho phép 2 nhánh nhặt song song để kiểm tra khả năng chọn nhánh tối ưu của tác tử; và C3 là bản đồ 8x8 với chuỗi 4 vật phẩm, trong đó vật phẩm $C$ bị khóa, buộc tác tử phải suy luận nhiều chặng thông qua công tắc hoặc ngọc bỏ qua."*

---

### Slide 9: Thuật toán 1: Policy Iteration (Model-based DP)
- **Thời lượng:** ~50 giây
- **Lời thoại:**
  > *"Thuật toán đầu tiên là Policy Iteration. Đây là phương pháp Quy hoạch động áp dụng khi ta biết trước mô hình môi trường $P$ và $R$. Thuật toán lặp xen kẽ giữa Đánh giá chính sách bằng phương trình Bellman kỳ vọng và Cải tiến chính sách tham lam cho đến khi hội tụ. Trong đề tài, Policy Iteration đóng vai trò là **nghiệm chuẩn tối ưu tuyệt đối (Ground Truth)** để đối chiếu với các thuật toán học tự do."*

---

### Slide 10: Thuật toán 2: Q-Learning (Model-free Off-policy TD)
- **Thời lượng:** ~50 giây
- **Lời thoại:**
  > *"Thuật toán thứ hai là Q-Learning. Đây là phương pháp học không cần mô hình, tác tử tự tương tác và cập nhật bảng $Q(s, a)$ dựa trên sai số Temporal-Difference. Với cơ chế Off-policy sử dụng $\max_{a'} Q(s', a')$ và chiến lược $\epsilon$-greedy suy giảm dần từ $1.0$ về $0.05$, tác tử đã tự động thám hiểm mê cung, mò mẫm ra chuỗi nhặt đồ tối ưu và đạt tỷ lệ chiến thắng 100% trên cả 3 bản đồ."*

---

### Slide 11: Thuật toán 3: SARSA (Model-free On-policy TD)
- **Thời lượng:** ~45 giây
- **Lời thoại:**
  > *"Thuật toán thứ ba là SARSA – đại diện cho trường phái On-policy TD control. Khác với sự lạc quan của Q-Learning, SARSA cập nhật dựa trên hành động thực tế tiếp theo $a'$. Do tính đến cả rủi ro thám hiểm ngẫu nhiên của bản thân, SARSA thể hiện tính cẩn trọng cao hơn, có xu hướng đi tránh xa các ô nguy hiểm và các ô nhặt sai thứ tự."*

---

### Slide 12: Kết quả thực nghiệm đối chuẩn hiệu năng
- **Thời lượng:** ~45 giây
- **Lời thoại:**
  > *"Đây là bảng tổng kết kết quả đánh giá độc lập 200 runs của 3 thuật toán trên 3 bản đồ. Quý Thầy Cô có thể thấy: Cả Policy Iteration, Q-Learning và SARSA đều đạt **tỷ lệ thành công 100.0% tuyệt đối**. Đặc biệt, số bước đi trung bình của Q-Learning khi khai thác (8 bước trên C1, 16 bước trên C2, 36 bước trên C3) trùng khớp hoàn toàn với số bước tối ưu lý thuyết của Policy Iteration."*

---

### Slide 13: Trực quan hóa quỹ đạo tối ưu trên bản đồ 2D
- **Thời lượng:** ~45 giây
- **Lời thoại:**
  > *"Trên màn hình là hình ảnh trực quan hóa quỹ đạo di chuyển tối ưu của tác tử. Tại C1, tác tử đi một mạch qua A, B, C rồi về Exit. Tại C2, tác tử tự động nhận biết nhánh bên trái ngắn hơn nhánh bên phải nên chọn đi nhánh 1. Tại C3, tác tử thực hiện một hành trình phức tạp: nhặt A, B, rẽ xuống gạt công tắc W để giải phóng C, nhặt C, D rồi về Exit an toàn trong đúng 36 bước."*

---

### Slide 14: Tổng kết đồ án và bài học kinh nghiệm
- **Thời lượng:** ~45 giây
- **Lời thoại:**
  > *"Tóm lại, đồ án đã hoàn thành xuất sắc toàn bộ mục tiêu đề ra: Mô hình hóa thành công bài toán ràng buộc thứ tự cứng bằng MDP 5 chiều; thiết kế hàm thưởng cân bằng triệt tiêu bẫy thoát non; và đối chuẩn toàn diện 3 thuật toán Tabular RL đạt độ chính xác 100%. Bài học kinh nghiệm lớn nhất rút ra là: Trong Học Tăng Cường, việc thiết kế không gian trạng thái thỏa mãn tính chất Markov và định hình hàm thưởng đúng đắn là yếu tố quyết định thành bại của mô hình. Em xin trân trọng cảm ơn quý Thầy Cô và Hội đồng đã lắng nghe!"*
