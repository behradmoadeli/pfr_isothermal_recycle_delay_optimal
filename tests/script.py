import os

folder_path = '/Users/behradmoadeli/Documents/PhD/Behrads_Papers/functions'  # Replace with the actual folder path

for filename in os.listdir(folder_path):
    if filename.endswith('.py'):  # Adjust the file extension as needed
        x = os.path.splitext(filename)[0]
        print(f"from .{x} import {x}")
