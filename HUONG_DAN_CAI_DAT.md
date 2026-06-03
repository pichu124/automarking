# Hướng dẫn cài đặt & chạy ở nhà (Windows)

Checklist để cài app trên máy mới (desktop ở nhà). Lệnh dành cho **PowerShell**.

---

## 0. Cài phần mềm cần thiết (làm 1 lần)

1. **Git** — https://git-scm.com/download/win (cài mặc định).
2. **Python 3.11+** — https://www.python.org/downloads/
   ⚠️ Khi cài, **TÍCH "Add Python to PATH"**.
3. **Ollama** (nếu chạy local) — https://ollama.com/download
   Hoặc bằng PowerShell: `irm https://ollama.com/install.ps1 | iex`

Kiểm tra:
```powershell
git --version
python --version
ollama --version
```

---

## 1. Tải code về

```powershell
cd $HOME
git clone https://github.com/pichu124/automarking.git
cd automarking
git checkout claude/blissful-hawking-ROFtm
```

Nếu sau này có bản cập nhật, chỉ cần:
```powershell
git pull origin claude/blissful-hawking-ROFtm
```

---

## 2. Cài thư viện Python

```powershell
pip install -r requirements.txt
```
(Nếu muốn gọn gàng có thể tạo venv, nhưng cài thẳng như trên là chạy được.)

---

## 3. Chọn cách chấm

### Cách A — Ollama (miễn phí, chạy local, riêng tư)

**3A.1. Kiểm tra GPU** (quyết định chọn model nào):
```powershell
nvidia-smi          # nếu có GPU NVIDIA -> xem dòng "Memory" (VRAM)
```
Không có lệnh này = không có GPU NVIDIA → sẽ chạy CPU (chậm).

**3A.2. Chọn model theo VRAM:**

| VRAM GPU | Model nên dùng | Lệnh pull |
|---|---|---|
| Không có GPU / ~4GB | `qwen2.5:3b` | `ollama pull qwen2.5:3b` |
| ~6–8GB | `qwen2.5:7b` | `ollama pull qwen2.5:7b` |
| ~12GB | `qwen2.5:14b` | `ollama pull qwen2.5:14b` |
| ~24GB+ | `qwen2.5:32b` | `ollama pull qwen2.5:32b` |

> `qwen2.5` bám rubric và xuất JSON tốt hơn llama cho việc chấm. Model càng lớn → chấm càng sắc nhưng cần nhiều VRAM.

**3A.3. Đặt context window** (giữ vừa đủ, đừng quá lớn):
```powershell
[System.Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH","20480","User")
```
Rồi **Quit Ollama ở khay hệ thống và mở lại** (để nhận giá trị mới).

> 20480 đủ cho 1 request (~16k token) + phần model viết ra. Nếu đã nén bài mẫu và rút gọn task thì có thể để 16384.

**3A.4. Sau khi chạy, kiểm tra GPU đang gánh bao nhiêu:**
```powershell
ollama ps
```
Cột `PROCESSOR` càng nghiêng về **GPU** (vd `20%/80% CPU/GPU`) → càng nhanh. Nếu vẫn `~100% CPU` → đổi model nhỏ hơn hoặc hạ context.

### Cách B — Gemini (nhanh, gần như miễn phí, chất lượng tốt)

1. Lấy API key miễn phí tại **https://aistudio.google.com** → *Get API key*.
2. Tạo file `.env` và điền key:
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```
   Sửa dòng: `GEMINI_API_KEY=AIza...key-cua-ban...` (lưu, đóng).
3. Trong app chọn provider **Google Gemini (free tier)**.

---

## 4. Chạy app

```powershell
streamlit run app.py
```
Trình duyệt mở `http://localhost:8501`.

---

## 5. Quy trình trong app

1. **Sidebar → Provider**: chọn Ollama (gõ tên model đã pull) hoặc Gemini.
2. **New Grading Session** → đặt tên.
3. **Mục 1**: upload **Task**, **Rubric**, **4 bài mẫu** (HD/DI/AVG/PASS).
4. Bấm **🔍 Structure rubric into components** → thấy bảng tiêu chí.
5. (Ollama) Bấm **🧭 Read & compress samples** → đợi nén bài mẫu → xem lại "Review anchor profiles".
6. **Sidebar → 📏 Document length**: Student report = `0` (đọc đủ bài), Reference sample = `0` (đã dùng hồ sơ nén).
7. **Mục 2**: upload toàn bộ bài sinh viên (hoặc 1 file `.zip`).
8. **Mục 3 → 📊 Estimate**: xem "One request ≈ ? tokens" phải **< OLLAMA_CONTEXT_LENGTH**. Nếu vượt → hạ Document length hoặc rút gọn task.
9. Bấm **🚀 Run Two-Pass Grading** → xong tải file **.docx**.

---

## 6. Tối ưu tốc độ Ollama (nếu chậm)

- Đổi model **nhỏ hơn** (qwen2.5:3b nhanh nhất).
- **Hạ** `OLLAMA_CONTEXT_LENGTH` (16384–20480) để model lọt GPU.
- Rút gọn ô **Task text** (bỏ trang bìa, hình phạt, mục lục…) — giảm token/request.
- Đảm bảo đã **nén bài mẫu** (bước 5).
- `ollama ps` để xác nhận GPU đang gánh phần lớn.
- Bài **đầu tiên luôn chậm** (nạp model) — bài sau nhanh hơn.

> Cần nhanh cho cả lớp mà máy yếu → dùng **Gemini free** (Cách B).
