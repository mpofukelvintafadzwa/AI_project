import os
@tool
def save_code(filename, code):

    with open(filename, "w") as f:
        f.write(code)
