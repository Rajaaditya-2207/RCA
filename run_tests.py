import subprocess, sys, os

result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
    capture_output=True, text=True, encoding='utf-8',
    cwd=r'e:\Aditya\RCA'
)

output = result.stdout + "\n---STDERR---\n" + result.stderr
out_path = os.path.join(r'e:\Aditya\RCA', 'test_results.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"Written to {out_path}, length={len(output)}")
