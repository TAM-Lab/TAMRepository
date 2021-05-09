import requests

def translate(content):
    content = content
    try:
        res = requests.get("http://shuyantech.com/api/cndbpedia/avpair?q=" + content).json()
        lis = res["ret"]
    except Exception as e:
        return "error"
    result = "error"
    for li in lis:
        if li[0] == "外文名称":
            result = li[1]
            break
    return result

if __name__ == "__main__":
    print(translate("血糖"))