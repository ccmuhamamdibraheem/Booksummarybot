import pyautogui
import time
import random
import string
 
pyautogui.FAILSAFE = False  
 
def y():
    while True:
        try:
            time.sleep(15)
            r = random.choice(string.ascii_lowercase)
            pyautogui.press(r)
        except pyautogui.FailSafeException:
            print("Fail-safe triggered")
 
y()