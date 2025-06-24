import tkinter as tk
from tkinter import scrolledtext
import threading
from product_chatbot import get_ai_response  # This is your assistant logic

def on_button_click():
    chat_display.insert(tk.END, "🗣 Listening...\n")
    chat_display.see(tk.END)
    talk_button.config(state=tk.DISABLED)
    threading.Thread(target=run_ai_response).start()

def run_ai_response():
    try:
        response = get_ai_response()
        chat_display.insert(tk.END, f"🤖 {response}\n\n")
    except Exception as e:
        chat_display.insert(tk.END, f"⚠️ Error: {e}\n\n")
    finally:
        chat_display.see(tk.END)
        talk_button.config(state=tk.NORMAL)

# 🎨 Set up the window
root = tk.Tk()
root.title("Voice Assistant")
root.geometry("600x500")
root.configure(bg="#f0f0f0")  # Light background
root.resizable(False, False)

# 🎙 Button first (so it's visible above the chat)
talk_button = tk.Button(
    root,
    text="🎙 Talk to Assistant",
    command=on_button_click,
    font=("Segoe UI", 12, "bold"),
    bg="#0078D7",
    fg="white",
    padx=10,
    pady=5
)
talk_button.pack(pady=(20, 10))  # Add some top margin

# 💬 Chat display below the button
chat_display = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Segoe UI", 11),
    bg="white",
    fg="black"
)
chat_display.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

root.mainloop()
