import urllib3
from bs4 import BeautifulSoup

def isChinese(s):
	if ord(s) > 0x4e00 and ord(s) < 0x9fff:
		return True
	elif ord(s)==44 or ord(s)==65292:#44->cosma in english;65292->cosma in mandarin
		return True
	else:
		return False

def _filter(s,kws):
	for kw in kws:
		if(kw in s):
			return False
	return True

def delete(s,kws):
	for kw in kws:
		s=s.replace(kw,"")
	return s

def extract_text(url):
	http = urllib3.PoolManager()
	r = http.request("GET", url)
	#r=http.request("GET", "http://l50740.pixnet.net/blog/post/43592029")
	soup = BeautifulSoup(r.data,"html.parser")
	span=soup.find_all('span')
	word_delete=["標楷體","新細明體","微軟正黑體"]
	word_delete_s=["粉絲專頁","粉絲團","歡迎","按讚","人氣","留言列表","關閉視窗","電話","住址","網址"]
	text=''
	for line in span:
		line_str=line.__str__()
		if(_filter(line_str,word_delete_s)):#if sentence contains any of the word in "word_delete_s",discard the whole sentence
			line_str=delete(line_str,word_delete)# delete garbage word
			for ch in line_str:
				if isChinese(ch):
					text=text+ch
	return text

if __name__=="__main__":
	import sys
	print(extract_text(str(sys.argv[1])))
