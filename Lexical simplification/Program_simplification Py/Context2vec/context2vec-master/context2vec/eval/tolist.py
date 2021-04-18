import csv
file = open("C:/Users/song123/Desktop/can.txt",'w+')
with open("C:/Users/song123/Desktop/candidates.txt", 'r') as f:
    sentence=f.readlines()

    #print(f.readlines())
    #for line in f.readlines():
    #    print(line)
    #    file.write(line)