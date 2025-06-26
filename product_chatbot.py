import subprocess
from gtts import gTTS

def text_to_speech(text, output_audio="user_audio.wav"):
    tts = gTTS(text)
    tts.save(output_audio)
    return output_audio

def generate_lip_sync(face_video, audio_file, output_path="result.mp4"):
    command = [
        "python", "Wav2Lip/inference.py",
        "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth",
        "--face", face_video,
        "--audio", audio_file,
        "--outfile", output_path
    ]
    subprocess.run(command, check=True)
    return output_path
