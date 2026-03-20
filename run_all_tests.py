import subprocess, sys

result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
    capture_output=True, text=True, encoding='utf-8',
    cwd=r'e:\Aditya\RCA'
)

with open(r'e:\Aditya\RCA\test_all_results.txt', 'w', encoding='utf-8') as f:
    f.write(result.stdout)
    f.write("\n---STDERR---\n")
    f.write(result.stderr)

print("Done. Return code:", result.returncode)
