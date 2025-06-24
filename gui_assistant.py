import tkinter as tk
from tkinter import scrolledtext
import threading
from product_chatbot import get_ai_response

def on_button_click():
    chat_display.insert(tk.END, "🗣 Listening...\n")
    chat_display.see(tk.END)
    talk_button.config(state=tk.DISABLED)
    threading.Thread(target=run_assistant).start()

def run_assistant():
    try:
        reply = get_ai_response()
        chat_display.insert(tk.END, f"🤖 {reply}\n\n")
    except Exception as e:
        chat_display.insert(tk.END, f"⚠️ Error: {e}\n\n")
    finally:
        chat_display.see(tk.END)
        talk_button.config(state=tk.NORMAL)

# Setup GUI window
root = tk.Tk()
root.title("Talking AI Assistant")
root.geometry("600x500")
root.configure(bg="#f4f4f4")
root.resizable(False, False)

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
talk_button.pack(pady=(20, 10))

chat_display = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Segoe UI", 11),
    bg="white",
    fg="black"
)
chat_display.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

root.mainloop()
