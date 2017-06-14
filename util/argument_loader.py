def import_arguments(args):
    dataset = args[1]
    if args[2] == "1":
        query_flag = False
    else:
        query_flag = True
    if len(args) <= 3:
        return dataset, query_flag, args[0] , 500001, 500001, 0, 0
    else:
        savename = args[3] 
        steps_1 = int(args[4])
        steps_2 = int(args[5])
        steps_3 = int(args[6])
        expand_flag = int(args[7])
        return dataset, query_flag, savename, steps_1, steps_2, steps_3, expand_flag
