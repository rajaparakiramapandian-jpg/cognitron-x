
import io

filename = 'c:/Users/21bca/OneDrive/Desktop/COGNITRONX/login_page.py'

with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Indentation check for lines 195-210:")
for i, line in enumerate(lines):
    if 194 <= i <= 209:
        line_repr = repr(line)
        print(f"Line {i+1}: {line_repr}")
