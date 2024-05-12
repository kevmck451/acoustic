import subprocess

def sound_alert(text):
    subprocess.run(['say', text])



subprocess.run(['say', "I'M DONE TRAINING BITCH, HIT ME UP AGAIN!"])