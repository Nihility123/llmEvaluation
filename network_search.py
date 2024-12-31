import re
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
import random
from urllib.parse import urlencode, urlunparse

ua = UserAgent()
headers = {
    'User-Agent': ua.random,
}
print(headers)

def set_random_agent():
    headers['User-Agent'] = ua.random

def bing_search(keyword, res_num):
    params1 = {
        'q': keyword,
    }
    encoded_params1 = urlencode(params1)
    # 构建完整的URL，元组
    url1 = urlunparse(('https', 'cn.bing.com', '/search', '', encoded_params1, ''))
    print(url1)

    try:
        # 创建会话对象
        session = requests.Session()
        response1 = session.get(url1, headers=headers, timeout=10)
        response1.raise_for_status()
    except:
        print("========== bing连接失败 ==========")
        return []

    soup = BeautifulSoup(response1.text, 'html.parser')
    results = soup.select('.b_algo')
    print(f"========== bing连接成功，找到{len(results)}条，使用前{res_num}条 ==========")
    pages = []

    for result in results[:res_num]:
        try:
            link_element = result.select('a')[1]
            href = link_element['href']
            title = link_element.text

            abstract_element = result.select_one('.b_caption p')
            abstract = abstract_element.get_text() if abstract_element else ''

            pages.append({'href': href, 'title': title, 'abstract': abstract})
        except Exception as e:
            print("++++++++++ 格式错误 ++++++++++")
            print(e)
            print("++++++++++++++++++++")

    return pages

def google_search(keyword, res_num):
    params = {'q': keyword + suffix}
    encoded_params = urlencode(params)
    # 构建完整的URL，元组
    url = urlunparse(('https', 'www.google.com', '/search', '', encoded_params, ''))
    print(url)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except:
        print("========== google连接失败 ==========")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.select('.N54PNb.BToiNc')
    print(f"========== google连接成功，找到{len(results)}条，使用前{res_num}条 ==========")
    pages = []

    for result in results[:res_num]:
        try:
            link_element = result.select('a')[0]
            href = link_element['href']
            title_element = link_element.select_one('h3') if link_element else ''
            title = title_element.get_text() if title_element else ''
            abstract = ""

            pages.append({'href': href, 'title': title, 'abstract': abstract})
        except Exception as e:
            print("++++++++++ 格式错误 ++++++++++")
            print(e)
            print("++++++++++++++++++++")
        
    return pages

def page_scratch(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding
        response.raise_for_status()
    except Exception as e:
        print(f"========== url <{url}> 连接失败 ==========")
        print(e)
        return ""

    soup = BeautifulSoup(response.text, 'html.parser')

    text = soup.get_text()
    text = re.sub("^\n+|\n+$", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)

    return text.replace("\n", "").replace("\r", "")

def get_text(page_list):
    texts = []
    for page in page_list:
        texts.append(page_scratch(page["href"]))
    return texts

def add_text(page_list):
    for page in page_list:
        page["text"] = page_scratch(page["href"])
    return page_list

def search(keyword, by=0, search_num=20):
    set_random_agent()
    print(headers)
    match by:
        case 0:
            pages = bing_search(keyword, search_num)
        case 1:
            pages = google_search(keyword, search_num)
        case _:
            pages = []
    return get_text(pages)


