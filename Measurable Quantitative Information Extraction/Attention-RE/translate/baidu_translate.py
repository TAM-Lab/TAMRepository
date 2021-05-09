import http.client
import hashlib
import urllib
import random
import json


def translate(q):
    appid = ''  # 填写你的appid
    secretKey = ''  # 填写你的密钥
    httpClient = None
    myurl = '/api/trans/vip/fieldtranslate'
    fromLang = 'zh'  # 原文语种
    toLang = 'en'  # 译文语种
    salt = random.randint(32768, 65536)
    domain = 'medicine'
    sign = appid + q + str(salt) + domain + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&domain=' + domain + '&sign=' + sign
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)["trans_result"][0]["dst"]
        return result
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


def main():
    result = translate("")
    print(result)


if __name__ == "__main__":
    main()
