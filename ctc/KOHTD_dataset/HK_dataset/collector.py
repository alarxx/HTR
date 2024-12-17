import os
import json

def merge_json_files(input_directory, output_file):
    all_data = []  # Список для хранения данных из всех JSON-файлов

    # Проходимся по всем файлам в директории
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):  # Проверяем, что файл имеет расширение .json
            file_path = os.path.join(input_directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)  # Загружаем содержимое JSON-файла
                    all_data.append(data)  # Добавляем данные в общий список
            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")

    # Записываем объединенные данные в выходной файл
    try:
        with open(output_file, 'w', encoding='utf-8') as output:
            json.dump(all_data, output, ensure_ascii=False, indent=4)
        print(f"Объединенный JSON успешно сохранен в {output_file}")
    except Exception as e:
        print(f"Ошибка при сохранении объединенного файла: {e}")

# Параметры
input_directory = "./ann/"  # Укажите путь к директории с JSON-файлами
output_file = "merged_annotation.json"  # Имя выходного файла

# Выполняем функцию
merge_json_files(input_directory, output_file)
