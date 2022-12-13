import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import json

# csv to json

    
def csv_to_json(csvfilename, jsonfilename, fieldnames):
    csvfile = open(csvfilename, 'r')
    jsonfile = open(jsonfilename, 'w')

    reader = csv.DictReader(csvfile, fieldnames)

    for row in reader:
        json.dump(row, jsonfile)
        jsonfile.write('\n')


# 학습 데이터를 csv에서 json으로 변환
csv_to_json('train_data.csv', 'train_data.json',
            ("_0", "_1", "_2", "_3", "_4", "_5", "_6",
             "_7", "label", "title", "content", "_8")
)

# 학습 데이터 불러오기
DB = []
with open('train_data.json') as j:
    DB = list(map(json.loads, j))

# 학습 데이터에서 특수 문자 제거
for v in DB:
    v["content"] = re.sub(r'[^a-zA-Z0-9 ]', '', v["content"].lower()).strip()

# 학습 데이터에서 falsy한 값 제거
DB = list(filter(lambda x: x["content"] != '' and x["content"] != None, DB))

# 학습 데이터에서 중복되는 값 제거
DB = [dict(t) for t in {tuple(d.items()) for d in DB}]

# label을 0과 1로 바꾸기
DB_is = []
for v in DB:
    DB_is.append(1 if v['label'] == "Real" else 0)