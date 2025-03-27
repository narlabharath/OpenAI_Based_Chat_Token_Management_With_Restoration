# Chatbot UI - Codespaces Setup 🚀

This project is designed to run in **GitHub Codespaces** with **Streamlit**. Follow these steps to set up and run the chatbot.

## 📂 Setup & Run Instructions

### 1️⃣ Open the Codespace
Start your **GitHub Codespace**. Once inside the terminal:

### 2️⃣ Activate the Virtual Environment  
If the virtual environment (`venv/`) already exists, **activate it**:

```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows
```

> ⚠️ **If the above fails**, the `venv` might be missing. Create it using:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
Ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Chatbot  
Start the **Streamlit** app in Codespaces:

```bash
streamlit run chatbot_ui_v2.py --server.port 8000 --server.address 0.0.0.0
```

Once running, open the **forwarded port** in your Codespace to access the UI.

---

## 🔄 Making & Pushing Changes  

If you make any updates, commit and push them to GitHub:

```bash
git add .
git commit -m "Updated chatbot"
git push origin main
```

---

## 🛠 Troubleshooting  

### ⚠️ Virtual Environment Not Found  
If `source venv/bin/activate` fails, recreate it:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### ⚠️ Streamlit Not Found  
If `streamlit` is missing after activation:
```bash
pip install streamlit
```

### ⚠️ Port Not Accessible  
Make sure the **Codespace ports are forwarded properly** (Port `8000`).

---

## 📌 Notes
- **The virtual environment should persist**, but if Codespaces resets, you may need to recreate it.
- Always activate the environment before running the app.
- If issues persist, restart the Codespace or reinstall dependencies.

---

🚀 **You're all set! Start chatting with the AI now.** 🎉

