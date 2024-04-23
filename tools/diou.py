import numpy as np
from iou import bb_intersection_over_union

def compute_diou(bboxA, bboxB,img_w,img_h):

    # 输入为非归一化坐标，转换为归一化坐标计算归一化距离
    x1, y1, x2, y2 = bboxA
    x1g, y1g, x2g, y2g = bboxB

    x1 = x1/img_w
    x2 = x2/img_w
    y1 = y1/img_h
    y2 = y2/img_h

    x1g = x1g/img_w
    x2g = x2g/img_w
    y1g = y1g/img_h
    y2g = y2g/img_h

    x2 = max(x1, x2)
    y2 = max(y1, y2)

    '''两个边界框的中心点'''
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2


    xc1 = min(x1, x1g)
    yc1 = min(y1, y1g)
    xc2 = max(x2, x2g)
    yc2 = max(y2, y2g)

    '''the intersection of bboxA and bboxB'''
    _,iouk,_,_ = bb_intersection_over_union(bboxA,bboxB)

    '''
    Rdiou=d2/c2
    c是覆盖这两个盒子的最小的盒子的对角线长度;
    d = ρ(b, b gt)是两个盒子中心点的距离
    '''
    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c

    '''DIoU-NMS'''
    diouk = iouk - u
    # print("iouk",iouk,"u",u,"diouk",diouk)

    return diouk

# if __name__ =="__main__":
#         bboxA = torch.tensor([1., 1., 6., 6.])
#         bboxB = torch.tensor([2., 2., 5., 5.])
#         ans = compute_diou(bboxA,bboxB)
#         print(ans)