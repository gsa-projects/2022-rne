import json

def make_unit(label: int, category: list, content: str):
    return {
        "label": label,
        "category": category,
        "content": content
    }

# load
DB = []
with open('./example.json') as j:
    DB = json.load(j)

# write
DB.append(make_unit(0, ['사회'], "sample1"))
DB.append(make_unit(2, ['테크', '과학'], "sample2"))
DB.append(make_unit(3, ['정치'], "sample3"))
DB.append(make_unit(1, ['경제'], "sample4"))

# save
with open('./example.json', 'w', encoding='utf-8') as file:
    json.dump(DB, file, ensure_ascii=False, indent='\t')