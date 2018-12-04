import os
import re
def extract_bike(folder="data/stanford40/XMLAnnotations"):
    ret = []
    for i in os.listdir(folder):
        with open(folder + "/" + i) as f:
            bike = []
            lines = f.readlines()
            for idx, j in enumerate(lines):

                j = j.strip()
                #print(j)
                if j == "<bndbox>":
                    bike.append(int(lines[idx + 1].strip()[6:-7]))
                    bike.append(int(lines[idx + 2].strip()[6:-7]))
                    bike.append(int(lines[idx + 3].strip()[6:-7]))
                    bike.append(int(lines[idx + 4].strip()[6:-7]))
            print(bike)
            ret.append(bike)
    #print (ret)

extract_bike()