with open('test.txt') as f:
    lines = [line for line in f.readlines()]

new_lines = []
for l in lines:
    new_lines.append(' '.join(l.split()[1:]))

with open('test_final.txt', 'w') as f:
    f.writelines('\n'.join(new_lines))