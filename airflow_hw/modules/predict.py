import dill
import pandas as pd
import os

# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '.')

# Укажем путь к директории с pkl-файлами обученных моделей
models_path = f'{path}/data/models'
# Получение списка имен pkl-файлов по заданному пути
model_filenames = os.listdir(models_path)

# Укажем путь к директории с json-файлами данных для обучения
test_data_path = f'{path}/data/test'
# Получение списка имен json-файлов по заданному пути
test_data_filenames = os.listdir(test_data_path)


# Получение последнего созданного pkl-файлами с обученной моделью
def get_model_filename():
    return max(model_filenames)


# Чтение pipeline-а с подготовкой данных и обученной моделью из pkl-файла
def load_model():
    name = get_model_filename()
    # Чтение сериализованного файла с пайплайном модели и её метаданными
    with open(models_path + '/' + name, 'rb') as file:
        model = dill.load(file)
    return model


def predict():
    result_df = None

    # Итерация по списку имен json-файлов с тестовыми данными
    for test_data_filename in test_data_filenames:
        # Чтение json-файла с тестовыми данными в Series
        series = pd.read_json(test_data_path + '/' + test_data_filename, typ='series')
        # Преобразование Series в DataFrame
        df = pd.DataFrame([series])
        # Предсказание
        model = load_model()
        y = model.predict(df)
        # Создание DataFrame с предсказанными значениями
        predict_df = pd.DataFrame({
            'id': df.id,
            'pred': y[0],
            'price': df.price
        })
        # Объединение DataFrame-ов с предсказанными значениями в один
        if result_df is None:
            result_df = predict_df
        else:
            result_df = pd.concat([result_df, predict_df], axis=0, ignore_index=True)

    # Сохранение в директорию data/predictions csv-файла
    model_filename = get_model_filename()
    preds_filename = f'preds_{model_filename.split("_")[-1].split(".")[0]}.csv'
    result_df.to_csv(f'{path}/data/predictions/{preds_filename}', index=False)


if __name__ == '__main__':
    predict()
