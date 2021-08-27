class PersonBody(object):
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

