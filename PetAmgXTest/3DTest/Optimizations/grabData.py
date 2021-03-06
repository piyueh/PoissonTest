import os
import numpy

cases = ["1GPU_Opt_NxNyNz160",
        "2GPU_Opt_NxNyNz160",
        "4GPU_Opt_NxNyNz160",
        "8GPU_Opt_NxNyNz160",
        "16GPU_Opt_NxNyNz160",
        "1GPU_Opt_Nx320Ny160Nz80",
        "2GPU_Opt_Nx320Ny160Nz80",
        "4GPU_Opt_Nx320Ny160Nz80",
        "8GPU_Opt_Nx320Ny160Nz80",
        "16GPU_Opt_Nx320Ny160Nz80",
        "1GPU_Opt_NxNyNz160_K40"]

#cases = ["1GPU_Opt_NxNyNz160",
#        "2GPU_Opt_NxNyNz160",
#        "4GPU_Opt_NxNyNz160",
#        "8GPU_Opt_NxNyNz160",
#        "16GPU_Opt_NxNyNz160"]
#cases = ["1GPU_Opt_NxNyNz160"]

slurmFiles = {case:[] for case in cases}
runCount = {case:{} for case in cases}
wallTime = {case:{} for case in cases}
bestSubCase = {}

keyword1 = "Case Name: "
keyword2 = "Solve Time:  "

for case in cases:
    for root, dirs, files in os.walk(case):
        for file in files:
            if file.endswith(".out") and file.find("slurm") != -1:
                slurmFiles[case].append(file)

for case in cases:
    for file in slurmFiles[case]:
        f = open(case + "/" + file, "r")
        for line in f:
            if line.find(keyword1) != -1:
                n1 = line.find(keyword1)
                n2 = line.find("\n")
                subCase = line[n1+len(keyword1):n2]
                if subCase in runCount[case]:
                    runCount[case][subCase] += 1
                else:
                    runCount[case][subCase] = 1
            if line.find(keyword2) != -1:
                n1 = line.find(keyword2)
                n2 = line.find("s wall")
                time = float(line[n1+len(keyword2):n2])
                if subCase in wallTime[case]:
                    wallTime[case][subCase] += time
                else:
                    wallTime[case][subCase] = time
        f.close()

    for subCase in wallTime[case].keys():
        wallTime[case][subCase] /= runCount[case][subCase]

    bestSubCase[case] = min(wallTime[case], key=wallTime[case].get)

for case in cases:
    print(case, ": ", bestSubCase[case], " ", 
            wallTime[case][bestSubCase[case]], " sec ",
            runCount[case][bestSubCase[case]], " runs")
