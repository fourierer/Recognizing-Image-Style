#此脚本用于分析测试集当中不同类别里面的错分样本，错分为哪些类别，且错分量为多少
#在划分好的测试数据集下（test_result_to_other）运行该脚本

import os

classlist = os.listdir('./test_result_to_other/')
num_class = len(classlist)
for i in range(num_class):
    category = classlist[i]
    print('large category:' + category)
    error_path = './test_result_to_other/' + category +'/wrong/'
    category_error_list = os.listdir(error_path)
    num_error_class = len(category_error_list)
    for j in range(num_error_class):
        category1 = category_error_list[j]
        error_category_path = './test_result_to_other/' + category + '/wrong/' + category1 + '/'
        image_list = os.listdir(error_category_path)
        error_num = len(image_list)
        print('category:' + category1 + '   num:' + str(error_num))



