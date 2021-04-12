# Лабораторная работа #1
## Подготовка окружения для решения задачи классификации изображений из набора данных Food-101 с использованием нейронных сетей глубокого обучения
### Изначальные результаты обучения
Оранжевая кривая отображает обучающую выборку, синяя - валидационную.
https://tensorboard.dev/experiment/HYYXpm51TnGWiqnVPPriqQ/#scalars&runSelectionState=eyJ0cmFpbiI6dHJ1ZSwidmFsaWRhdGlvbiI6dHJ1ZX0%3D
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi-lab1/main/epoch_categorical_accuracy%20(1).svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi-lab1/main/epoch_loss%20(1).svg">

## Создать и обучить сверточную нейронную сеть произвольной архитектуры с количеством сверточных слоев >3.

### 1. Сверточная нейронная сеть организована из функции Conv2D, функций Гаусса GaussianNoise и GaussianDropout, функции MaxPool2D.
```
 inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.GaussianNoise(0.1)(x)
  x = tf.keras.layers.GaussianDropout(0.1)(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.GaussianNoise(0.5)(x)
  x = tf.keras.layers.GaussianDropout(0.5)(x)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
 ```
https://tensorboard.dev/experiment/dRp9nqMfQS2uEYtPi7I8xg/#scalars&runSelectionState=eyJ0cmFpbiI6dHJ1ZSwidmFsaWRhdGlvbiI6dHJ1ZX0%3D         
#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi-lab1/main/epoch_categorical_accuracy%20(2).svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi-lab1/main/epoch_loss%20(2).svg">

## Анализ результатов
