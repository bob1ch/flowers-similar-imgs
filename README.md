# Поиск похожих картинок (цветов)

0. Перед запуском программы нужно получить kaggle.json(API) на личной странице и положить рядом с app.py

	0.1 Это нужно для того, чтобы программа могла подтянуть датасеты для работы

	0.2 Придется подождать пока догрузятся датасеты, извините :(

1. Linux ```source venv/bin/activate```

	1.1 Если windows ```python -m venv venv```

	1.2```venv\Scripts\activate.bat```

	1.3```pip install -r requirements.txt```

2. Запуск приложения ```flask --app app run```
3. Загрузите картинку с компьютера
4. Нажмите кнопку "загрузить"
5. Нажмите кнопку "найти похожие"

В ноутбуке "Обучение и тест инференса.ipynb"  содержится код для обучения и там же я тестировал инференс модели

В качестве базовой модели был выбран VGG19 с кастомной головой

В дальнейшем эта модель файнтюнилась на данных по цветам

Для реализации поиска картинок было задано 2 output: 
1. Отвечал за классификацию изображения, 
2. возвращал карту признаков