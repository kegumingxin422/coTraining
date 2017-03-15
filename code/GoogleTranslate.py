#!/usr/bin/python
# -*- coding: utf-8 -*-
#Python -V: Python 2.6.6
#filename:GoogleTranslation1.2.py

__author__ = "Yinlong Zhao (zhaoyl[at]sjtu[dot]edu[dot]cn)"  
__date__ = "$Date: 2013/04/21 $" 

import re
import urllib,urllib2

#urllib:
#urllib2: The urllib2 module defines functions and classes which help in opening
#URLs (mostly HTTP) in a complex world — basic and digest authentication,
#redirections, cookies and more.


def translate(text, f, t):
    
    '''模拟浏览器的行为，向Google Translate的主页发送数据，然后抓取翻译结果 '''
    
    #text 输入要翻译的英文句子
    text_1=text
    #'langpair':'en'|'zh-CN'从英语到简体中文
    #values={'hl':'zh-CN','ie':'UTF-8','text':text_1,'langpair':"'en'|'zh-CN'"}
	values = {'hl':'zh-CN','ie':'UTF-8','text':text,'langpair':"%s|%s"%(f, t)}
    url='http://translate.google.cn'
    data = urllib.urlencode(values)
    req = urllib2.Request(url,data)
    #模拟一个浏览器
    browser='Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 2.0.50727)'
    req.add_header('User-Agent',browser)
    #向谷歌翻译发送请求
    response = urllib2.urlopen(req)
    #读取返回页面
    html=response.read()
    #从返回页面中过滤出翻译后的文本
    #使用正则表达式匹配
    #翻译后的文本是'TRANSLATED_TEXT='等号后面的内容
    #.*? non-greedy or minimal fashion
    #(?<=...)Matches if the current position in the string is preceded
    #by a match for ... that ends at the current position
    filename='html.txt'
    fp=open(filename,'w')
    fp.write(html)
   # print html
    p=re.compile(r"(?<=TRANSLATED_TEXT=).*?;")
    m=p.search(html)
    text_2=m.group(0).strip(';')
    return text_2

if __name__ == "__main__":
    #text_1 原文
    text_1='Hello, my name is Derek. Nice to meet you! '
    print('The input text: %s' % text_1)
    text_2=translate(text_1, 'en', 'zh-CN').strip("'")
    print('The output text: %s' % text_2)


