import matplotlib.pyplot as plt

charint = {
    ">": 21,
    "0": 9,
    "6": 15,
    "{": 24,
    "(": 4,
    "1": 10,
    "7": 16,
    "5": 14,
    "#": 1,
    '"': 0,
    ")": 5,
    "*": 6,
    "+": 7,
    "}": 25,
    "4": 13,
    "-": 8,
    ":": 19,
    "2": 11,
    "3": 12,
    "^": 22,
    "9": 18,
    "8": 17,
    "&": 3,
    "$": 2,
    "_": 23,
    "=": 20,
}

intchar = {
    0: "23",
    1: "10",
    2: "11",
    3: "14",
    4: "16",
    5: "17",
    6: "15",
    7: "19",
    8: "-",
    9: "0",
    10: "1",
    11: "2",
    12: "3",
    13: "4",
    14: "5",
    15: "6",
    16: "7",
    17: "8",
    18: "9",
    19: "24",
    20: "20",
    21: "12",
    22: "13",
    23: "18",
    24: "22",
    25: "21",
}
dist = {
    "0": 17288,
    "1": 680322,
    "2": 266871,
    "3": 281662,
    "4": 120511,
    "5": 194594,
    "6": 248714,
    "7": 105843,
    "8": 55968221,
    "9": 4346985,
    "10": 1270243,
    "11": 3488223,
    "12": 3034658,
    "13": 1709770,
    "14": 2511833,
    "15": 883409,
    "16": 1918110,
    "17": 746225,
    "18": 840976,
    "19": 35828,
    "20": 70870,
    "21": 569052,
    "22": 215254,
    "23": 72396,
    "24": 62915,
    "25": 45225,
}
x = []
guide = []
nums = []
for i in range(26):
    nums.append(intchar[i])
    x.append(dist[str(i)])
    guide.append(i)
print(nums)
print(x)
plt.plot(guide, x)
# for num in dist:
#    x.append(intchar[int(num)])
#    xp.append(num)
#    y.append(dist[num])
#
# print(x)
# print(y)
