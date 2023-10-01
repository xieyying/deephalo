
file1=r"C:\Users\Administrator\Desktop\training_result\NPatlas_200_1600\config.toml"
file2=r"C:\Users\Administrator\Desktop\training_result\NPatlas_200_1600\config_edited.toml"

with open(file1,'r') as f1:
    #将f1中的0 = 1 行替换为0 = 2，将f1中的1 = 1 行替换为1 = 2，将f1中的2 = 1 行替换为2 = 2，将f1中的3 = 1 行替换为3 = 1
    lines = f1.readlines()
line_nw = []
for i in range(len(lines)):
    if lines[i].startswith("0 = 1"):
        lines[i] = "0 = 2\n"
        line_nw.append(lines[i])
    elif lines[i].startswith("1 = 1"):
        lines[i] = "1 = 2\n"
        line_nw.append(lines[i])
    elif lines[i].startswith("2 = 1"):
        lines[i] = "2 = 2\n"
        line_nw.append(lines[i])
    elif lines[i].startswith("3 = 1"):
        lines[i] = "3 = 1\n"
        line_nw.append(lines[i])
    else:
        line_nw.append(lines[i])
with open(file2,'w') as f2:
        for l in line_nw:
            f2.write(l)


