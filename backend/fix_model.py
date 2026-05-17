with open('lgbm_stream1.txt', 'r') as f:
    lines = f.readlines()

if lines[0].strip() == 'tree':
    lines = lines[1:]

if not lines[0].startswith('pandas_categorical'):
    lines.insert(0, 'pandas_categorical:null\n')

with open('lgbm_stream1_fix.txt', 'w') as f:
    f.writelines(lines)
