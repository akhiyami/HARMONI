import subprocess

scripts = [
    "evaluation/memory/llm_as_a_judge.py",
    "evaluation/memory/llm_as_a_judge_ltm.py",
    "evaluation/memory/llm_as_a_judge_lt_stm.py"
]

for script in scripts:
    print(f"\n➡️ Running {script}...\n")
    # This will stream output live to the console
    result = subprocess.run(["python", script])
    
    if result.returncode == 0:
        print(f"\n✅ Finished {script}\n")
    else:
        print(f"\n❌ {script} failed with exit code {result.returncode}\n")
        # Uncomment the next line if you want to stop on failure:
        # break


