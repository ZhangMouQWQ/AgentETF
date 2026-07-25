"""清缓存 + 运行 main.py, 输出保存到文件"""
import os, sys, shutil, subprocess

ROOT = r'c:\Users\Administrator\Documents\AgentETF'

# 1. 清 __pycache__
for root, dirs, files in os.walk(ROOT):
    if '__pycache__' in dirs:
        p = os.path.join(root, '__pycache__')
        shutil.rmtree(p, True)
        print(f'[CLEAN] {p}')

# 2. 运行 main.py
python_exe = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
main_py = os.path.join(ROOT, 'main.py')

print(f'\n[RUN] {main_py}')
result = subprocess.run([python_exe, main_py], cwd=ROOT,
                        capture_output=True, text=True, timeout=600)

# 3. 保存输出
with open(os.path.join(ROOT, '_run_out.txt'), 'w', encoding='utf-8') as f:
    f.write(result.stdout)
    if result.stderr:
        f.write('\n\n=== STDERR ===\n')
        f.write(result.stderr)

print(f'[EXIT] code={result.returncode}')
print(f'[OUTPUT] {len(result.stdout)} chars, {len(result.stderr)} chars stderr')
