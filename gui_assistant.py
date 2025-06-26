import os
from product_chatbot import text_to_speech, generate_lip_sync

def main():
    print("Type a sentence and press Enter (type 'exit' to quit)")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        if not user_input:
            continue

        print("Converting to speech...")
        audio_path = text_to_speech(user_input)

        print("Generating video...")
        result_path = generate_lip_sync("input_face.mp4", audio_path)

        if os.path.exists(result_path):
            print("✅ Video ready! Opening...")
            os.startfile(result_path)
        else:
            print("⚠️ Video generation failed.")

if __name__ == "__main__":
    main()
