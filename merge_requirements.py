def merge_requirements(conda_file, pip_file, output_file):
    with open(conda_file, "r") as file:
        conda_lines = file.readlines()

    with open(pip_file, "r") as file:
        pip_lines = file.readlines()

    with open(output_file, "w") as file:
        for line in conda_lines:
            if not line.startswith("_") and "=" in line:
                package = line.split("=")
                if len(package) >= 2:
                    file.write(f"{package[0]}=={package[1]}\n")

        for line in pip_lines:
            file.write(line)


if __name__ == "__main__":
    merge_requirements(
        "conda_requirements.txt", "pip_requirements.txt", "requirements.txt"
    )
