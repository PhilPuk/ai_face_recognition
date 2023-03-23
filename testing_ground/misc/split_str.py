import os
name = "C:/Student/Documents/User.1.5.jpg"

new = os.path.split(name)[-1].split(".")[1]

print(new)