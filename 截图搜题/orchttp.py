import requests
import base64
import json
import os

host = 'https://ocrapi-ugc.taobao.com'
path = '/ocrservice/ugc'
appcode = '15c23fa577a24ee18a73000168185527'
url = host + path

# bodys[''] = "{//图像数据：base64编码，要求base64编码后大小不超过4M，最短边至少15px，最长边最大4096px，支持jpg/png/bmp格式，和url参数只能同时存在一个\"img\":\"\",//图像url地址：图片完整URL，URL长度不超过1024字节，URL对应的图片base64编码后大小不超过4M，最短边至少15px，最长边最大4096px，支持jpg/png/bmp格式，和img参数只能同时存在一个\"url\":\"\",//是否需要识别结果中每一行的置信度，默认不需要。true：需要false：不需要\"prob\":false}"
# post_data = bodys['']
# request = urllib2.Request(url, post_data)
# request.add_header('Authorization', 'APPCODE ' + appcode)
# # //根据API的要求，定义相对应的Content-Type
# request.add_header('Content-Type', 'application/json; charset=UTF-8')
# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode = ssl.CERT_NONE
# response = urllib2.urlopen(request, context=ctx)
# content = response.read()
# if (content):
#     print(content)
headers = {
    'Authorization': 'APPCODE ' + appcode,
    'Content-Type': 'application/json; charset=UTF-8'
}


def orc(im):
    with open(im, 'rb')as f:
        im = f.read()
        byte64 = base64.b64encode(im)
        s = str(byte64, encoding='utf-8')
        response = requests.post(url, json={'img': s, 'prob': False}, headers=headers)
        json_str = response.text
        # print(json_str)
        data = json.loads(json_str)
        # print(repr(data))
        dics = data['prism_wordsInfo']
        # print(type(dics))
        # 图片文本
        q = ''
        for i in dics:
            q = q + i['word']
        print(q)
        os.system('echo {}| clip'.format(q))