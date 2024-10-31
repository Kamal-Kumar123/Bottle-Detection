# # In starting during MNNIT internship on computer vision I worked on spyder ide (Anaconda)
# # I just paste this here in vscode
# # -*- coding: utf-8 -*-
# """
# Created on Sun Sep  8 22:08:00 2024

# @author: KAMAL KUMAR
# """

import cv2
import numpy as np
import pyautogui

# Get the screen dimensions
screen_width, screen_height = pyautogui.size()
screen_size = (screen_width, screen_height)

# Define codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
path = input("provide path where recording will be saved:$$ ")

output = cv2.VideoWriter(path, fourcc, 20.0, screen_size)

print("Press 'q' to stop recording or Ctrl+C to exit.")
try:
    while True:
        # Take a screenshot
        img = pyautogui.screenshot()
        
        # Convert to a NumPy array and switch RGB to BGR for OpenCV
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video file
        output.write(frame)

        # (Optional) Display the live recording
        cv2.imshow("Screen Recorder", frame)
        
        # Stop recording if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nRecording stopped by user.")

# Release resources
output.release()
cv2.destroyAllWindows()
