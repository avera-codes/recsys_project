import os

# Чтение списка необходимых библиотек
with open("requirements.txt") as f:
    required_packages = {line.strip().split("==")[0] for line in f}

# Получение списка всех установленных пакетов
installed_packages = os.popen("conda list --name recsys --export").read().splitlines()

# Формат списка пакетов: имя=версия=канал
installed_packages = {pkg.split("=")[0] for pkg in installed_packages}

# Определение пакетов, которые нужно удалить
packages_to_remove = installed_packages - required_packages

# Удаление пакетов
for pkg in packages_to_remove:
    os.system(f"conda remove --name recsys --yes {pkg}")
