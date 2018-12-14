from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir("raw_data") if isfile(join("raw_data", f))]

for file in onlyfiles:
    f = open("raw_data/" + file)
    print(f)