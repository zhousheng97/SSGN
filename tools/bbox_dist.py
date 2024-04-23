'''
求两个矩形之间的最小距离:
(1) 不相交，但在X或Y轴方向上有部分重合坐标，比如矩形1和2，此时，最小距离为两个矩形之间的最小平行距离或垂直距离，如图中红色箭线D12所示。
(2) 不相交，在X和Y轴方向上均无重合坐标，比如矩形1和3，此时最小距离为两矩形距离最近的两个顶点之间的距离，如图中红色箭线D13。
(3) 相交，此时距离为负数，如矩形1和4。
'''

import math

def min_distance_of_bbox(bboxA,bboxB):

    '''首先计算两个矩形中心点'''
    bboxA_xmin, bboxA_ymin, bboxA_xmax, bboxA_ymax = bboxA[:]
    bboxA_center_x = 0.5 * (bboxA_xmin + bboxA_xmax)
    bboxA_center_y = 0.5 * (bboxA_ymin + bboxA_ymax)

    bboxB_xmin, bboxB_ymin, bboxB_xmax, bboxB_ymax = bboxB[:]
    bboxB_center_x = 0.5 * (bboxB_xmin + bboxB_xmax)
    bboxB_center_y = 0.5 * (bboxB_ymin + bboxB_ymax)

    '''分别计算两矩形中心点在X轴和Y轴方向的距离'''
    Dx = abs(bboxA_center_x - bboxB_center_x)
    Dy = abs(bboxA_center_y - bboxB_center_y)

    bbA_w = bboxA_xmax - bboxA_xmin 
    bbA_h = bboxA_ymax - bboxA_ymin 
    bbB_w = bboxB_xmax - bboxB_xmin 
    bbB_h = bboxB_ymax - bboxB_ymin 

    '''
    两矩形不相交，在X轴方向有部分重合的两个矩形，
    最小距离是上矩形的下边线与下矩形的上边线之间的距离

    两矩形不相交，在Y轴方向有部分重合的两个矩形，
    最小距离是左矩形的右边线与右矩形的左边线之间的距离
    
    两矩形不相交，在X轴和Y轴方向无重合的两个矩形，最小距离是距离最近的两个顶点之间的距离，
	利用勾股定理，很容易算出这一距离
    '''
    if (Dx<(bbA_w+bbB_w)/2) & (Dy>=(bbA_h+bbB_h)/2):
        min_dist = Dy-(bbA_h+bbB_h)/2
    elif (Dx>=(bbA_w+bbB_w)/2) & (Dy<(bbA_h+bbB_h)/2):
        min_dist = Dx-(bbA_h + bbB_h)/2
    elif (Dx>=(bbA_w+bbB_w)/2) & (Dy>=(bbA_h+bbB_h)/2):
        delta_x = Dx - (bbA_w+bbB_w)/2
        delta_y = Dy - (bbA_h+bbB_h)/2
        min_dist = math.sqrt(delta_x * delta_x + delta_y * delta_y)
    else:
            min_dist = -1
    return min_dist



