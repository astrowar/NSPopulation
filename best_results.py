# localiza os melhores resultados de cada modelo ( 2,3,4 gaussianas)

import os
import re

rx = "[-+]?\d+\.\d+(?:[Ee][+-]?\d+)?"


class Model:
    def __init__(self, uuid, mo, xo, so, loss, bic):
        self.uuid = uuid
        self.bic = bic
        self.loss = loss
        self.so = so
        self.xo = xo
        self.mo = mo
        self.ordem = len(xo)
        print(xo, mo, so , loss, bic)

    def __repr__(self):
        return self.uuid + ":" + str(self.ordem) + " LOSS:"+str(self.loss)


def get_files(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
    return files


def to_array(s):
    q = re.findall(rx, s)
    return [float(x) for x in q]


# 90f37b4d-9503-4cfa-b876-8d5ba532c791

def get_models(filename):
    models = []
    uuid = ""
    xo = []
    so = []
    mo = []
    lk = []
    bic = []

    for s in open(filename).readlines():
        s = s.strip("\n")
        s = s.strip(' ')
        if re.match("\w+-\w+-\w+-\w+-\w+", s):
            # print(uuid)
            if (uuid != ""):
                models.append(Model(uuid, mo, xo, so, lk, bic))
            uuid = s.strip('\n')
            xo = []
            so = []
            mo = []
            lk = []
            bic = []

        if s.startswith("x:"): xo = to_array(s)
        if s.startswith("s:"): so = to_array(s)
        if s.startswith("m:"): mo = to_array(s)
        if s.startswith("loss"): lk = float(re.findall(rx, s)[0])
        if s.startswith("BIC"): bic = float(re.findall(rx, s)[0])

    return models


files = get_files("results")
models = []
[models.extend(get_models(f)) for f in files]
for x in models:
    print(x)
