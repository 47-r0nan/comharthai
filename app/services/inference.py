import pyttsx3


# Dummy model logic for now — replace with actual model prediction later
def predict_letter(img):
    return "T"


def speak_letter(letter):
    engine = pyttsx3.init()
    engine.say(letter)
    engine.runAndWait()
