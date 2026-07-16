# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 18:34:14 2022

@author: youm
"""
#srun --pty --time=1-0 --mem=64G --gres=gpu:1 --partition=gpu bash -i
# Y:\  ==  /home/groups/graylab_share/Chin_Lab/ChinData
#ssh youm@arc-infra-3

import os

#PATH= r"C:\Users\youm\Desktop"
#PATH = r"\\graylab\BCC_Chin_Lab_RDS\ChinData"   #"W:\ChinData"
#PATH = r"\\graylab.ohsu.edu\Chin_Lab"
#PATH = r"\\graylab\share\chinkoei"    #dataxchange"
#PATH = "C:/"
#PATH = r"\\graylab\share\engje"
#PATH = r"\\graylab\Chin_Lab\_Images\AxioScan2"
#PATH = r'\\graylab\Chin_Lab\_Images\AxioScan2.Old'
#PATH = r'\\graylab\Chin_Lab\_Images'   #\AxioScan_2016_Marathon'
#PATH = r'Y:'
#PATH = r'C:\Users\youm\Desktop\src'
#GNS = ['AxioScan2']
'''
local format
PATHS = [r'\\graylab\Chin_Lab\_Images\AxioScan2',r'\\graylab\Chin_Lab\_Images\AxioScan2.Old',
         r'\\graylab\Chin_Lab\Cyclic_Workflow',r'\\graylab\Chin_Lab\Cyclic_Workflow.Old',
         r'\\graylab\Chin_Lab\Cyclic_Analysis',r'\\graylab\Chin_Lab\Cyclic_Analysis.Old',
         r'\\graylab\Chin_Lab\Cyclic_Images']

'''
PATHS = [r'/home/groups/graylab_share/Chin_Lab/ChinData/_Images/AxioScan2',r'/home/groups/graylab_share/Chin_Lab/ChinData/_Images/AxioScan2.Old',
         r'/home/groups/graylab_share/Chin_Lab/ChinData/Cyclic_Workflow',r'/home/groups/graylab_share/Chin_Lab/ChinData/Cyclic_Workflow.Old',
         r'/home/groups/graylab_share/Chin_Lab/ChinData/Cyclic_Analysis',r'/home/groups/graylab_share/Chin_Lab/ChinData/Cyclic_Analysis.Old',
         r'/home/groups/graylab_share/Chin_Lab/ChinData/Cyclic_Images']

PATHS = [r'C:\Users\youm']
UNMEASURED = []

LEVELS = 10

def main():
    for PATH in PATHS:
        outName = PATH.split("\ "[0])[-1].split("/")[-1]+'_new'
        print(outName)
        out = dig(PATH)
        #print(out,len(out))
        print(out)
        out = parse(out)
        print(out,'after parse')
        for i in range(len(out)):
            out[i] = out[i][2:]
        #for o in out:
            #print(o)
        #return()

        f = open(outName+"full-rws.csv","w")
        for line in out:
            if len(line.split(',')) > LEVELS+2: #4 is 2 lvls, 3 is 1 lvl, 5 is 3 etc
                continue
            try:
                f.write(line)
            except Exception as e:
                try:
                    print("could not print",line,"because",e)
                except:
                    print("error2",e)
        f.close()
        f = open("UNMEASURED_"+outName+".csv","w")
        f.writelines(UNMEASURED)
        f.close()



def parse(lis):
    out = []
    i = 0
    while i < len(lis):
        #print(i)
        elem = lis[i]
        if type(elem) == list:
            pe = parse(elem)
            #print(pe,"pe")
            for ne in pe:
                out.append(" ,"+ne)
            i += 1
        else:
            #print(elem,"elem")
            out.append(lis[i]+","+str(round(lis[i+1]/1099511627776,2))+ "\n")
            i += 2
    #print(out,"out")
    return(out)


def dig(path):
    out = []
    #print(path,'path')
    if os.path.isdir(path):

        total = 0
        name = path.split("/")[-1].split("\ "[0])[-1]
        if name == "_Images" or name == "_Temp" or "Cyclic_" in name:
            print("\n\n              !!!!!!!!!!!!!!!!")
        #if name == "recovered files":
            #return([["recovered files",1000000000]])
        print(name)
        try:
            for f in sorted(os.listdir(path)):
                if os.path.isdir(path+"/"+f):
                    layer = dig(path+"/"+f)
                    total += scrapeLists(layer)
                    out.append(layer)
                else:
                    try:
                        total += os.path.getsize(path+"/"+f)
                    except Exception as e:
                        print("*******           COULD NOT GET SIZE OF",f)
                        UNMEASURED.append(path+"/"+f+" "+str(e)+" file\n")
        except Exception as e:
            print("....    COULD NOT ENTER FOLDER",path)
            UNMEASURED.append(path+" "+str(e)+" file\n")

        out.insert(0,[name,total])
    print(out,"dig out")
    return(out)



def scrapeLists0(layer):
    total = 0
    for l in layer:
        if type(l) == float or type(l) == int:
            total += l
        else:
            pass
            #print(l,"scraping")
    #print("sl total",total)
    return(total)

def scrapeLists(layer):
    total = 0
    for l in layer:
        if type(l) == list:
            total += scrapeLists0(l)
        elif type(l) == float or type(l) == int:
            total += l
        else:
            pass
            #print(l,"scraping")
    #print("sl total",total)
    return(total)


def scrapeLists1(layer):
    total = 0
    sublists = 0
    for l in layer:
        if type(l) == list:
            sublists += 1
    if sublists == 0:
        for l in layer:
            if type(l) == float or type(l) == int:
                total += l
    else:
        for l in layer:
           if type(l) == list:
               total += scrapeLists1(l)
            #print(l,"scraping")
    #print("sl total",total)
    print(layer,total)
    return(total)


main()