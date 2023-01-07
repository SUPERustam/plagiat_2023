# AntiPlagiat (Tinkoff '23)

```bash
$ ./compare.py -h

usage: AntiPlagiat [-h] [-p] [-a] [-n] [-r] [-i] [-d] [-s] [-t] file1 file2

Send me two txt files: first' ' with list of filepaths to check for plagiat an' ' second to save result of my work

positional arguments:
  file1                 txt file of filepaths with this format:"file plagiat_file" etc.
  file2                 txt file for results. Previous data in file will be removed ❌

optional arguments:
  -h, --help            show this help message and exit
  -p, --pure            Don't pre-process files. Without -p flag I will use my cool ast-base pre-process technologies
  -a, --advanced        Use Damerau–Levenshtein distance instead of Levenshtein distance
  -n, --nice-view       Show result in persents instead of fractions
  -r , --round-number   Round result
  -i , --insertion      Edit insertion cost. Default 1.0
  -d , --deletion       Edit deletion cost. Default 1.0
  -s , --substitution   Edit substitution cost. Default 1.0
  -t , --trans          Edit transposition cost. Default 1.0. Only for -a flag

You can also combine flags e.g. '-na' equal to '-n -a'
```
## Примеры работы программы
Примеры запросов
```sh
python compare.py input.txt score.txt
python compare.py input.txt score.txt -pa
python compare.py input.txt score.txt -a -t 0.5 --nice-view
python compare.py another_input.txt score2.txt -r 3 -n -d 1.2 -i 0.3
```
Вызов программы
```sh
python compare.py
# или
python3 compare.py
# или
./compare.py
```
Пример файла для входных данных `file1` (`.txt` тип файла)
```plain
files/main.py plagiat1/main.py
files/loss.py plagiat2/loss.py
files/loss.py files/loss.py
```
Пример файла для выходных данных `file2` (`.txt` тип файла)
```plain  
0.3
0.843434
0.1533234234
```
## Установка
1. Python 3.9+
2. Терминал с поддержкой UTF8 и эмодзи
3. Библиотека `numpy`
```sh
pip install -r requirements.txt
```
## Что внутри
Весь код находится в `compare.py` так как такое было задание (еще можно было использовать `train.py` и `model.pkl`). Еще есть папки `files`, `plagiat1` и
`plagiat2` для тестов и пробного использования программы, `LICENSE` лицензия, `.gitignore` и `.git` для работы с `git`, `.vimspector.json` для дебагера 
vimspector, `requirements.txt` для внешних библиотек и конечно сам `README.md`
# Как работает приложение
### CLI интерфейс
Реализован с помощью `argparse`, который позволяет сделать приложение удобным и с богатым функционалом
### Препроцессинг
Обработка Python файлов c помощью `ast` и `re`. Удаление всех комментариев, декоративных элементов, пустых строк итд.

С флагом `-p` программа не проводит препроцессинг
### Расстояние Левенштейна
В базовом режиме используется эта [метрика](https://en.wikipedia.org/wiki/Levenshtein_distance), которая позволяет вычислить схожесть программ/текстов.
Вычисление этого расстояния, произоводится с помощью [алгоротма Вагнера-Фишера](https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm) 
с использования всего двух строк матрицы вместо n (количество символов во втором тексте). Пространственная сложность составляет O(m) вместо O(mn), где m и n 
длины текстов.

`0.0` - абсолютно разные тексты 

`1.0` - абсолютно одинаковые тексты
### Расстояние Дамерау — Левенштейна
Чтобы его вычислить добавьте флаг `-a`.

Почти то же самое, что и предыдущая метрика, только учитывает еще транспозиции символов в тексте.
Пространственная сложность O(m).

`0.0` - абсолютно разные тексты

`1.0` - абсолютно одинаковые тексты
### Другие технологии
Программа умеет еще изменять цену каждой операции (удаление, замена, добавление и транспозиция), красиво выводить результат в разных форматах, 
записывать в файл ответы, а под копотом используется ООП, type hints, `numpy`, `ast`, `re`, принципы DRY, EAFP, LYBL и KISS. И конечно много алгоритмов!

