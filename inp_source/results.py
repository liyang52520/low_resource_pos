import os
import re

if __name__ == '__main__':
    for language in ["de", "en", "es", "fr", "id", "it", "ja", "ko", "pt", "sv"]:
        # for language in ["de"]:
        print("=" * 20, language, "=" * 20)
        devs = []
        tests = []
        for seed in range(5):
            for file in os.listdir(f"save/{language}_all_{seed}/"):
                if file.startswith("train-gaussian"):
                    break
            with open(os.path.join(f"save/{language}_all_{seed}/", file), encoding="utf8") as f:
                lines = f.readlines()
                # devs.append(re.findall("VM: (\d+\.\d+)%", lines[-3].strip())[0])
                tests.append(re.findall("VM: (\d+\.\d+)%", lines[-1].strip())[0])
        # print('\n'.join(devs))
        # print("=")
        print('\n'.join(tests))
        # print("\n")
