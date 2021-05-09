#################################################################################
# usage of the script
# usage: python search-terms.py -k APIKEY -v VERSION -s STRING
# see https://documentation.uts.nlm.nih.gov/rest/search/index.html for full docs
# on the /search endpoint
#################################################################################

from Authentication import *
import requests
import json

apikey = ""
version = "current"
string = "11"
uri = "https://uts-ws.nlm.nih.gov"
content_endpoint = "/rest/search/" + version
AuthClient = Authentication(apikey)
tgt = AuthClient.gettgt()
pageNumber = 0
ticket = AuthClient.getst(tgt)
pageNumber += 1
query = {'string': string, 'ticket': ticket, 'pageNumber': pageNumber}

r = requests.get(uri + content_endpoint, params=query)
r.encoding = 'utf-8'
items = json.loads(r.text)
jsonData = items["result"]
result = jsonData["results"][0]
try:
    uri = result["uri"].replace("2020AA", "current")
    # 根据得到的uri进行再一次的查找
    ticket = AuthClient.getst(tgt)
    query = {"ticket": ticket}
    r = requests.get(uri, params=query)
    r.encoding = 'utf-8'
    items = json.loads(r.text)
    jsonData = items["result"]
    category = jsonData["semanticTypes"][0]["name"]
    print(string, " ", category)
except Exception as e:
    print(string, " ", "error")
