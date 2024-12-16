# ./cmnist-master/allrgb/ -> ./CMNIST

import os
from os.path import join, getsize
import shutil


src_path = "./cmnist-master/allrgb/"
dst_path = "./CMNIST"

# Создаем целевой каталог, если его еще нет
# if exist_ok=False метод будет поднимать ошибку, если директория уже существует
os.makedirs(dst_path, exist_ok=True)

# os.scandir
# Получаем итератор объектов DirEntry для указанного каталога
# with os.scandir(src_path) as entries:
# 	print("entries:", entries)
# 	for entry in entries:
# 		print(f"entry:{entry},", 				end=" ")
# 		print(f"path:{entry.path},", 			end=" ")
# 		print(f"name:{entry.name}", 			end=" ")
# 		print(f"is_file():{entry.is_file()}", 	end=" ")
# 		print(f"is_dir():{entry.is_dir()}")

# Проходим по всем файлам в исходном каталоге
with os.scandir(src_path) as entries:
	print("entries:", entries)
	for entry in entries:
		if entry.is_file():  # Проверяем, что это файл
			# Извлекаем первую часть имени файла до символа "_"
			prefix = entry.name.split("_")[0]

			# Путь к подкаталогу для текущего префикса
			target_folder = os.path.join(dst_path, prefix)
			os.makedirs(target_folder, exist_ok=True)  # Создаем подкаталог, если его нет

			# Путь для копирования файла
			target_file_path = os.path.join(target_folder, entry.name)

			# Копируем файл в целевой подкаталог
			shutil.copy(entry.path, target_file_path)
			print(f"Copied {entry.name} to {target_folder}")
