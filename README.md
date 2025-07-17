# Распознавание действий на видео с использованием 3D-CNNS
Проект разработан для итоговой аттестационной работы по курсу «Профессия ML-инженер».
В работе использована модель R3D_18 на Data Set Kinetics-400 дообученная на data set UCF-101. 

## Оглавление
- [Технологии](#Технологии)
- [Использование](#Использование)
- [Deploy](#Deploy)
## Технологии
- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Gradio](https://www.gradio.app/)
## Использование
Клонируйте репозиторий на компьютер
### Обучение
- Подготовка данных и обучение модели в файле 3D_CNNS_zip.ipynb;
- Перед обучением загрузите датасет UCF-101 (https://www.crcv.ucf.edu/data/UCF101.php) в zip-архиве
### Запуск приложения
- Загрузите веса модели (https://drive.google.com/file/d/16MNA73JX7954z7PehDw_9A5qsbv0Vdzf/view?usp=sharing) в корневую директорию
- Для запуска выполните: 
```sh
python app.py
```
- Приложение будет доступно в браузере по адресу: http://127.0.0.1:7860
- Для выхода нажмите ctr+C
## Deploy
Посмотреть действующее приложение можно по ссылке: 
https://huggingface.co/spaces/aquarapax/total_work
## Ссылки


 



