from flask import Flask, jsonify, request
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import math
import time
from base64_func import *
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from scipy.spatial.distance import pdist, squareform


class Person_Body(object):
    def __init__(self, index, head_point: list, body_box: list, body_point: list):
        self.__index = index
        self.__head_point = head_point
        self.__body_box = body_box
        self.__body_point = body_point
        self.__flag = False
        self.__rate = 0
        self.__score = 1

    def set_score(self, score):
        self.__score = score

    def get_score(self):
        return self.__score

    def set_rate(self, rate):
        self.__rate = rate

    def get_rate(self):
        return self.__rate

    def get_body_area(self) -> int:
        w = self.__body_box[2] - self.__body_box[0]
        h = self.__body_box[3] - self.__body_box[1]
        return w * h

    def get_person_index(self) -> int:
        return self.__index

    def get_head_point(self) -> list:
        return self.__head_point

    def get_body_point(self) -> list:
        return self.__body_point

    def get_body_box(self) -> list:
        return self.__body_box

    def get_flag(self) -> bool:
        return self.__flag

    def set_flag(self, status: bool):
        self.__flag = status


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu and use_cuda:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


# 点对点欧拉距离
def p2p_euclidean(point1: list, point2: list) -> int:
    """
    :param point1:[x1,y1]
    :param point2:[x2,y2]
    :return:distance
    """
    offset_x = point1[0] - point2[0]
    offset_y = point1[1] - point2[1]
    return int(math.sqrt(offset_x ** 2 + offset_y ** 2))


# 求头部与所有安全帽的最小距离
def get_distance(head_points: list, hat_point: list) -> int:
    """
    :param point1: head point :[[x1,y1],[x2,y2],[y3,y3],...]
    :param points: hat points:[x1,y1]
    :return: min distance
    """

    min_dis = 99999
    for head in head_points:
        ret = p2p_euclidean(head, hat_point)
        if ret < min_dis:
            min_dis = ret
    return min_dis


# 判断在阈值范围内，是否有匹配的安全帽
def is_hat(person: Person_Body, hats_point: list, img_area: int) -> bool:
    # 头部平均高度
    if person.get_head_point() == []:
        return False
    # print("points:", person.get_head_point())
    np_points = np.array(person.get_head_point())
    print("head points:", np_points)
    Y = int(np.average(np_points, 0)[1])
    print("头部平均高度", Y)
    # 求人头坐标间的最大距离
    points = person.get_head_point()
    points = np.array(points)
    max_head_dis = round(np.max(squareform(pdist(points))))

    body_area = person.get_body_area()
    body_img_rate = body_area / img_area
    print("body_area:{},人体-图片占比:{}".format(body_area, body_img_rate))
    person.set_rate(body_img_rate)

    if body_img_rate > 0.4:
        print("0.8倍缩小")
        max_head_dis *= 0.8
    else:
        if body_img_rate < 0.001:
            return False
        print("1.2倍放大")
        max_head_dis *= 1.2
    print("头内坐标间最大距离:", max_head_dis)

    for hat in hats_point:
        # 安全帽在人体范围内
        if hat[1] > Y:
            print("安全帽高度:", hat[1])
            print("安全帽低于头部")
            continue
        # 获取头部和帽子最小距离
        ret = get_distance(person.get_head_point(), hat)
        print("头帽最小距离:", ret)
        if max_head_dis > ret:
            return True

    print("安全帽离开头部")
    return False


print("加载模型")
net = PoseEstimationWithMobileNet()
checkpoint_path = "./checkpoint_iter_370000.pth"
assert os.path.exists(checkpoint_path) == True, 'weight path is not exists'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
load_state(net, checkpoint)

# assert os.path.exists("./test_imgs") == True, 'test img file is not exists'
height_size = 256
cpu = False
track = 1
smooth = 1

net = net.eval()
use_cuda = torch.cuda.is_available()
print("cuda:", use_cuda)
if not cpu and use_cuda:
    print("cpu:", cpu)
    net = net.cuda()

print("加载模型完成")
stride = 8
upsample_ratio = 4
num_keypoints = Pose.num_kpts
delay = 1

app = Flask(__name__)

# 颜色表
RED = (0, 0, 255)
BULD = (255, 0, 0)
GREED = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
PURPLE = (160, 32, 240)
BLACK = (0, 0, 0)


@app.route('/get_head_point', methods=['POST'])
def get_head_point():
    """
      input:{"img":["img_base64","img_base64",...]}
      :return:{"location":[[[x,y],[x,y],...],[[x,y],[x,y],...],...]}
      """
    print("Function:get the head point")
    is_drwa = False

    params = request.json if request.method == "POST" else request.args
    imgs = base64_decode2cv2(params["img"])
    # for i,img in enumerate(imgs):
    #     cv2.imwrite("{}.jpg".format(i),img)

    location = []
    for img in imgs:
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        print("人体个数：", len(pose_entries))

        head_point = []
        for n in range(len(pose_entries)):
            head_point = []
            if len(pose_entries[n]) == 0:
                continue
            # pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            pose_keypoints = np.ones((num_keypoints, 2), dtype=int) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

            pose = Pose(pose_keypoints, pose_entries[n][18])
            # 耳朵关键点
            ears = [16, 17]
            ears_flag = True
            for ear in ears:
                # print("pose_keypoints[ear]:",pose_keypoints[ear])
                if pose_keypoints[ear][0] == -1 or pose_keypoints[ear][1] == -1:
                    ears_flag = False
                    continue
                # print('1:',type(pose_keypoints[ear].tolist()))

                temp_point = [int(pose_keypoints[ear][0]), int(pose_keypoints[ear][1])]
                head_point.append(temp_point)

            # 当耳朵关键点完整时推断中心点
            if ears_flag:
                offset = abs(pose_keypoints[17][0] - pose_keypoints[16][0])
                x = min(pose_keypoints[17][0], pose_keypoints[16][0])
                x += int(0.5 * offset)

                offset = abs(pose_keypoints[17][1] - pose_keypoints[16][1])
                y = min(pose_keypoints[17][1], pose_keypoints[16][1])
                y += int(0.5 * offset)

                new_point = [int(x), int(y)]
                if pose_keypoints[16][1] == pose_keypoints[17][1]:
                    new_point = (int(x), int(pose_keypoints[17][1]))

                print(type(new_point[0]), type(new_point[0]))
                head_point.append(new_point)

            face = [0, 14, 15]
            for f in face:
                if pose_keypoints[f][0] != -1 and pose_keypoints[f][1] != -1:
                    temp_point = [int(pose_keypoints[f][0]), int(pose_keypoints[f][1])]
                    head_point.append(temp_point)
            if is_drwa:
                # 绘制人体骨骼
                pose.draw(img)
                # 绘制头部关键点
                for p in head_point:
                    cv2.circle(img, tuple(p), 3, BULD, -1)
            location.append(head_point)
    return get_result("200", "Success", location)


@app.route('/get_head_point_v2', methods=['POST'])
def get_head_point_v2():
    """
      input:{"img":["img_base64","img_base64",...]}
      :return:{"location":[[[x,y],[x,y],...],[[x,y],[x,y],...],...]}
      """
    print("Function:get the head point")
    is_drwa = True

    params = request.json if request.method == "POST" else request.args
    imgs = base64_decode2cv2(params["img"])
    # for i,img in enumerate(imgs):
    #     cv2.imwrite("{}.jpg".format(i),img)

    location = []
    for img in imgs:
        head_point = []
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        print("人体个数：", len(pose_entries))

        for n in range(len(pose_entries)):
            head_point = []
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=int) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

            pose = Pose(pose_keypoints, pose_entries[n][18])
            # 耳朵关键点
            ears = [16, 17]
            ears_flag = True
            for ear in ears:
                # print("pose_keypoints[ear]:",pose_keypoints[ear])
                if pose_keypoints[ear][0] == -1 or pose_keypoints[ear][1] == -1:
                    ears_flag = False
                    continue
                # print('1:',type(pose_keypoints[ear].tolist()))

                temp_point = [int(pose_keypoints[ear][0]), int(pose_keypoints[ear][1])]
                head_point.append(temp_point)

            # 当耳朵关键点完整时推断中心点
            if ears_flag:
                offset = abs(pose_keypoints[17][0] - pose_keypoints[16][0])
                x = min(pose_keypoints[17][0], pose_keypoints[16][0])
                x += int(0.5 * offset)

                offset = abs(pose_keypoints[17][1] - pose_keypoints[16][1])
                y = min(pose_keypoints[17][1], pose_keypoints[16][1])
                y += int(0.5 * offset)

                new_point = [int(x), int(y)]
                if pose_keypoints[16][1] == pose_keypoints[17][1]:
                    new_point = (int(x), int(pose_keypoints[17][1]))

                print(type(new_point[0]), type(new_point[0]))
                head_point.append(new_point)

            face = [0, 14, 15]
            for f in face:
                if pose_keypoints[f][0] != -1 and pose_keypoints[f][1] != -1:
                    temp_point = [int(pose_keypoints[f][0]), int(pose_keypoints[f][1])]
                    head_point.append(temp_point)
            if is_drwa:
                # 绘制人体骨骼
                pose.draw(img)
                # 绘制头部关键点
                for p in head_point:
                    cv2.circle(img, tuple(p), 3, BULD, -1)
                cv2.imwrite("./img.jpg", img)
                cv2.waitKey(0)
            location.append(head_point)
    return get_result("200", "Success", location)


@app.route('/get_body_box', methods=['POST'])
def get_body_box():
    print("Function:get body box")
    is_drwa = False

    output = []
    person_list = []
    params = request.json if request.method == "POST" else request.args
    img = base64_decode2cv2(params["img"])
    img = img[0]

    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                 total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

    print("人体个数：", len(pose_entries))
    for n in range(len(pose_entries)):
        head_point = []
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

        pose = Pose(pose_keypoints, pose_entries[n][18])

        # 去除负数项
        ret = [i for i in pose_keypoints if i[0] > 0]
        np_pose_keypoints = np.array(ret)
        # print("ret:",ret)
        ret = np.min(np_pose_keypoints, 0)
        x1, y1 = ret
        ret = np.max(np_pose_keypoints, 0)
        x2, y2 = ret

        # 人体边框坐标列表
        body_list = []

        body_list.append(int(x1))
        body_list.append(int(y1))
        body_list.append(int(x2))
        body_list.append(int(y2))
        person_body = Person_Body(n, head_point, body_list, pose_keypoints)
        person_list.append(person_body)

    if not person_list:
        return get_result("200", "Empty", [])
    for person in person_list:
        temp_dic = {}
        box = person.get_body_box()
        x1, y1 = (box[0], box[1])
        x2, y2 = (box[2], box[3])

        # temp_dic["x1"] = x1
        # temp_dic["y1"] = y1
        # temp_dic["x2"] = x2
        # temp_dic["y2"] = y2

        temp_dic["height"] = y2 - y1
        temp_dic["left"] = x1
        temp_dic["top"] = y1
        temp_dic["width"] = x2 - x1

        temp_dic["rate"] = round(person.get_rate(), 5) * 100
        temp_dic["flag"] = person.get_flag()
        temp_dic['score'] = person.get_score()

        if is_drwa:
            if person.get_score():
                cv2.rectangle(img, (x1, y1), (x2, y2), GREED, 1)
        output.append(temp_dic)

    print("output:", output)
    if is_drwa:
        cv2.imwrite("./ret_imgs/ret_{}.jpg".format("test"), img)

    return get_result("200", "Success", output)


@app.route('/', methods=['POST'])
def human_pose():
    is_drwa = False

    output = []
    person_list = []
    params = request.json if request.method == "POST" else request.args
    img = base64_decode2cv2(params["img"])
    img = img[0]
    img_h, img_w, img_c = img.shape
    img_area = img_h * img_w
    print("img_w:{},img_h:{},img_area:{}".format(img_w, img_h, img_area))
    location = params["location"]
    alarm_type = params['alarmtype']

    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                 total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

    print("人体个数：", len(pose_entries))
    for n in range(len(pose_entries)):
        head_point = []
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

        pose = Pose(pose_keypoints, pose_entries[n][18])

        # 耳朵关键点
        ears = [16, 17]
        ears_flag = True
        for ear in ears:
            if pose_keypoints[ear][0] == -1 or pose_keypoints[ear][1] == -1:
                ears_flag = False
                continue
            head_point.append(pose_keypoints[ear])

        # 当耳朵关键点完整时推断中心点
        if ears_flag:
            offset = abs(pose_keypoints[17][0] - pose_keypoints[16][0])
            x = min(pose_keypoints[17][0], pose_keypoints[16][0])
            x += int(0.5 * offset)

            offset = abs(pose_keypoints[17][1] - pose_keypoints[16][1])
            y = min(pose_keypoints[17][1], pose_keypoints[16][1])
            y += int(0.5 * offset)

            new_point = [x, y]
            if pose_keypoints[16][1] == pose_keypoints[17][1]:
                new_point = (x, pose_keypoints[17][1])
            head_point.append(new_point)

        face = [0, 14, 15]
        for f in face:
            if pose_keypoints[f][0] != -1 and pose_keypoints[f][1] != -1:
                head_point.append(pose_keypoints[f])
        if is_drwa:
            # 绘制人体骨骼
            pose.draw(img)
            # 绘制头部关键点
            for p in head_point:
                cv2.circle(img, tuple(p), 3, BULD, -1)

        # 去除负数项
        ret = [i for i in pose_keypoints if i[0] > 0]
        np_pose_keypoints = np.array(ret)
        # print("ret:",ret)
        ret = np.min(np_pose_keypoints, 0)
        x1, y1 = ret
        ret = np.max(np_pose_keypoints, 0)
        x2, y2 = ret

        # 人体边框坐标列表
        body_list = []
        expand_rate_x = 0.2  # 边界扩展因子
        expand_rate_y = 0.1  # 边界扩展因子

        body_list.append(int(x1))
        body_list.append(int(y1))
        body_list.append(int(x2))
        body_list.append(int(y2))
        person_body = Person_Body(n, head_point, body_list, pose_keypoints)
        person_list.append(person_body)

    if alarm_type == 2:
        print("Alarm_type:", alarm_type)
        cloth_location_list = hltw2xxyy(location)
        print(cloth_location_list)
        for person in person_list:
            print("\n当前第{}个人".format(person.get_person_index()))
            is_cloth(person, cloth_location_list)

    if alarm_type == 3:
        print("Alarm_type:", alarm_type)
        # 获取安全帽中心点
        hat_centre_points = get_centre_point(location)
        print("安全帽个数:", len(hat_centre_points))

        if is_drwa:
            # 绘制安全帽中心点
            for i, p in enumerate(hat_centre_points):
                cv2.circle(img, tuple(p), 6, WHITE, -1)

        # 遍历匹配人体和安全帽
        for person in person_list:
            print("\n当前第{}个人".format(person.get_person_index()))
            ret = is_hat(person, hat_centre_points, img_area)
            print("是否戴安全帽：", ret)
            person.set_flag(ret)

    for person in person_list:
        temp_dic = {}
        box = person.get_body_box()
        x1, y1 = (box[0], box[1])
        x2, y2 = (box[2], box[3])

        temp_dic["x1"] = x1
        temp_dic["y1"] = y1
        temp_dic["x2"] = x2
        temp_dic["y2"] = y2
        temp_dic["rate"] = round(person.get_rate(), 5) * 100
        temp_dic["flag"] = person.get_flag()
        temp_dic['score'] = person.get_score()

        if is_drwa:
            if person.get_rate() < 0.001:
                continue
            if person.get_flag():
                cv2.rectangle(img, (x1, y1), (x2, y2), GREED, 1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), RED, 1)
        output.append(temp_dic)

    print("output:", output)
    if is_drwa:
        cv2.imwrite("./ret_imgs/ret_{}.jpg".format("test"), img)

    return get_result("200", "Success", output)


def hltw2xxyy(location: list) -> list:
    item_all = []
    for i in location:
        item = []
        item.append(i['left'])
        item.append(i['top'])
        item.append(i['left'] + i['width'])
        item.append(i['top'] + i['height'])
        item_all.append(item)
    return item_all


def mat_inter(box1: list, box2: list) -> bool:
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def solve_coincide(box1: list, box2: list) -> bool:
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    if mat_inter(box1, box2) is True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / (area1 + area2 - intersection)
        return coincide
    else:
        return False


def is_cloth(person: Person_Body, cloth_loction: list):
    rate = 0
    for cloth in cloth_loction:
        ret = solve_coincide(cloth, person.get_body_box())
        if ret > rate:
            rate = ret
    person.set_score(rate)
    if rate >= 0.4:
        person.set_flag(True)

    print("person i:{},rate:{}".format(person.get_person_index(), rate))


def get_centre_point(location: list) -> list:
    """
    :param location: [{left,top,width,height},{left,top,width,height}]
    :return: [[x,y],[x,y]]
    """
    point = []
    print(location)
    for part in location:
        part_point = []
        x = part['left'] + 0.5 * part['width']
        y = part['top'] + 0.5 * part['height']

        part_point.append(int(x))
        part_point.append(int(y))
        point.append(part_point)
    return point


# 构建接口返回结果
def get_result(code, message, data):
    result = {
        "code": code,
        "message": message,
        "data": data
    }
    print("Response data:", result)
    return jsonify(result)


app.config['JSON_AS_ASCII'] = False
app.run(host='0.0.0.0', port=24417, debug=False, use_reloader=False)
