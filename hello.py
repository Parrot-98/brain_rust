import json

x = []

# the formula is y = 2x
for i in range(100000):
    x.append(i * 2)

with open("data.json", "w") as f:
    json.dump(x, f)
