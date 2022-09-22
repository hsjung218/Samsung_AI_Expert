

def Correctness(output, target):
    correct = output.eq(target.view_as(output)).sum()
    accu = correct.float()
    return accu