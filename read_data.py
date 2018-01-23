class trace(object):
    def __init__(self, plain, key, mask, cipher):
        self.plain = plain
        self.key = key
        self.mask = mask
        self.cipher = cipher

    def getplain(self):
        return self.plain

    def getkey(self):
        return self.key

    def getmask(self):
        return self.mask

    def getcipher(self):
        return self.cipher


traceset = []
fileline = []
file = open("data.txt")
for line in file:
    fileline.append(line)
num_trace = int(len(fileline) / 5)
for i in range(num_trace):
    num = fileline[i * 5][:-1]
    plain = fileline[i * 5 + 1][:-1]
    key = fileline[i * 5 + 2][:-1]
    mask = fileline[i * 5 + 3][:-1]
    cipher = fileline[i * 5 + 4][:-1]
    tr = trace(plain, key, mask, cipher)
    traceset.append(tr)
