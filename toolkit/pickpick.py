import random
i = 0
pick = []
while i<2:
    row = random.randint(1,8)
    col = random.randint(1,4)
    #已参加班组织，方书隽，张卓宇，李娜，刘晨茜
    if (row == 6 and col == 1) or (row == 4 and col == 3) or \
         (row == 7 and col == 2) or (row == 7 and col == 3): 
        continue
    #非五团支部，卞若男,何梦,刘劲峰
    if (row == 1 and col == 4) or (row == 8 and col == 2) or (row == 4 and col == 1): 
        continue
    if row == 8 and col == 4: #空位
        continue
    i += 1
    pick.append((row,col))
print(pick)
