
cur = 0.5
for i in range(100):
    if(cur>0.5):
        operation = 'and'
        cur = cur*cur
    else:
        operation = 'or '
        cur = 1-( (1-cur) * (1-cur))
    print(i,operation, cur)


