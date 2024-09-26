import os
import asyncio
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import opendatasets as od #for kaggle download
from get_dataset import get_dataset_for_KNN
from sklearn.neighbors import NearestNeighbors
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)

# Папка для сохранения загруженных изображений и похожих
UPLOAD_FOLDER = 'static/uploads/'
SIMILAR_IMAGES_FOLDER = 'static/similar_images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SIMILAR_IMAGES_FOLDER'] = SIMILAR_IMAGES_FOLDER

CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'tulip',     'wild rose']
NAME = ''

# Допустимые форматы файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = [256, 256]
# Функция проверки расширения файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_datasets_from_kaggle():
    print('im here')
    if 'flowers-recognition' not in os.listdir():
        dataset = 'https://www.kaggle.com/datasets/alxmamaev/flowers-recognition'
        od.download(dataset)
    if 'tpu-getting-started' not in os.listdir():
        dataset = 'https://www.kaggle.com/c/tpu-getting-started/data'
        od.download(dataset)

def dummy_model(image_path):
    #грузим модель
    global NAME, CLASSES
    def load_model():
        base_model = tf.keras.applications.VGG19(
        weights='imagenet',
        include_top=False ,
        input_shape=[*IMAGE_SIZE, 3]
        )
        #base_model.trainable = True
        
        inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
        
        x = base_model(inputs, training=True)
        #x = tf.keras.layers.Flatten()(x)
        out2 = x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
        outputs = tf.keras.layers.Dense(104, activation='softmax')(x)
        model = tf.keras.Model(inputs, [outputs, out2])
        return model
    
    model = load_model()  # Определите функцию load_model отдельно
    model.load_weights('weights/weights.h5')

    # Загрузка изображения и его предобработка
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = np.array(image)

    # Получение предсказания
    y_pred = np.argmax(model(np.expand_dims(image, axis=0))[0])
    NAME = CLASSES[y_pred]
    ds_knn = get_dataset_for_KNN(y_pred)
    ds_knn_list = list(ds_knn.as_numpy_iterator())
    ds_knn_arr = np.array([item[0] for item in ds_knn_list])

    # Обучение KNN
    image_vectors = model.predict(ds_knn_arr)[1]
    knn = NearestNeighbors(metric='cosine', algorithm='auto')
    knn.fit(image_vectors)

    vec = model(np.expand_dims(image, axis=0))[1]
    dist, indices = knn.kneighbors(vec, n_neighbors=3)

    similar_imgs = ds_knn_arr[indices[0]]

    # Список для хранения путей к сохраненным изображениям
    similar_images_paths = []
    similar_imgs = [similar_imgs[0], similar_imgs[1], similar_imgs[2]]
    # Сохранение изображений на диск
    for idx, img in enumerate(similar_imgs):
        img_filename = f'similar_image_{idx}.jpeg'
        img_path = os.path.join(app.config['SIMILAR_IMAGES_FOLDER'], img_filename)

        # Сохраняем изображение
        img = (img * 255).astype(np.uint8)
        im = Image.fromarray(img)
        im.save(img_path)
        similar_images_paths.append(img_filename)
    return similar_images_paths

# Главная страница
@app.route('/', methods=['GET', 'POST'])
def index():
    get_datasets_from_kaggle()
    if request.method == 'POST':
        # Проверка наличия файла
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        # Если файл выбран и формат допустим
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # После загрузки перенаправляем на страницу с изображением
            return redirect(url_for('show_image', filename=filename))

    return render_template('index.html')

# Показ загруженного изображения
@app.route('/image/<filename>', methods=['GET', 'POST'])
def show_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    similar_images = []
    global NAME
    
    if request.method == 'POST':
        # Если была загружена новая картинка
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Перенаправляем на тот же маршрут для отображения нового результата
                return redirect(url_for('show_image', filename=filename))

        # Вызов модели для поиска похожих изображений
        similar_images = dummy_model(image_path)

    return render_template('show_image.html', filename=filename, similar_images=similar_images, flower_name=NAME)


if __name__ == '__main__':
    app.run(debug=True)
