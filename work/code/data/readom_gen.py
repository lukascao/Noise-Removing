import random

def get_upper():
    '''
    生成大写字母
    '''
    count = random.randint(1, 3)
    return random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=count)

def get_special_char():
    '''
    生成特殊符号
    '''

    count = random.randint(1, 3)

    return random.choices('!@$%^&*()_+~', k=count)

def get_lower(count):
    '''
    生成小写字母和数字
    '''

    string = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return random.choices(string, k=count)

def generate_password(length):
    if length < 6:
        length = 6
    lst = []
    upper_lst = get_upper() # 大写
    # special_char = get_special_char() # 特殊字符
    lst.extend(upper_lst)
    # lst.extend(special_char)
    surplus_count = length - len(lst)
    lower_lst = get_lower(surplus_count)
    lst.extend(lower_lst)
    # 将顺序打乱
    random.shuffle(lst)
    return ''.join(lst)

def Unicode(lenth):
    val_str = ''
    for _ in range(lenth):
        val = random.randint(0x4e00, 0x9fbf)
        val_str += chr(val)
    return val_str

def GBK2312(lenth):
    val_str = ''
    for _ in range(lenth):
        head = random.randint(0xb0, 0xf7)
        body = random.randint(0xa1, 0xfe)
        val = f'{head:x} {body:x}'
        # str = bytes.fromhex(val).decode('gb2312')
        str = bytes.fromhex(val).decode('gbk')
        val_str += str
    # str = bytes.fromhex(val).decode('gb2312')
    return val_str

def randomcolor():
    # get_colors = lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF)
    get_colors = lambda i: "#" + "%06x" % random.randint(1, 0xFFFFF0)
    return get_colors(1)

def generate_mark(lenth):
    # idx = random.choice([1, 1, 1])
    idx = random.choice([1, 3])
    if idx == 1:
        return Unicode(lenth)
    # if idx == 2:
    #     return GBK2312(lenth)
    if idx == 3:
        return generate_password(lenth)


if __name__ == '__main__':
    generate_mark(6)
