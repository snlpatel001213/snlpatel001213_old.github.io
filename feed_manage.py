
import feedparser
import json
import dateparser
import traceback
import datetime
from datetime import timedelta
import time


def get_feed_parsed(rss_feed_url):
    d = feedparser.parse(rss_feed_url)
    return d

def days_between(d1, d2):
    d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def get_relevant_fields(parsed_feed):
    feed_array = []
    for each_entry_no in range(len(parsed_feed.entries)):
        dict_ = {}
        try:
            # print("#################################################")
            feed_image = json.dumps(parsed_feed.entries[each_entry_no]['media_content'][0]['url'])
            dict_["feed_image"] = feed_image.replace('"',"") 
            # print(feed_image)
            feed_link = json.dumps(parsed_feed.entries[each_entry_no]['link'])
            dict_["feed_link"] = feed_link.replace('"',"")
            # print(dict_["feed_link"])
            feed_title = json.dumps(parsed_feed.entries[each_entry_no]['title'])
            dict_["feed_title"] = feed_title.replace('"',"") 
            # print(feed_title)
            feed_published = json.dumps(parsed_feed.entries[each_entry_no]['published'])
            date_parsed = dateparser.parse(feed_published.replace('"',""), settings={'TIMEZONE': 'UTC'}).strftime("%Y-%m-%d")

            dict_["feed_published"] = date_parsed
            current_date_time = dateparser.parse(str(datetime.datetime.now()), settings={'TIMEZONE': 'UTC'}).strftime("%Y-%m-%d")
    
            delta = days_between(current_date_time , date_parsed)
            dict_["day_difference"] = delta
            dict_["week_difference"] = int(delta/7)
            # FEED_POST_TRMPLATE = '<div class="chapter"><a href='+feed_link+'><img src='+feed_image+' alt='+feed_title+'></a><div class="chapter_inner"><p class="chapter_number"'+feed_published+'</p><a href='+feed_link+'s><h3 class="chapter_title">'+feed_title+'</h3></a></div></div>'
            # card_string = card_string + FEED_POST_TRMPLATE
            # print("#################################################")
            feed_array.append(dict_)
        except:
            ""
            # print(traceback.print_exc())
    return feed_array


def struct_time_corrector(string):
    new_date = ""
    string = string.replace("time.struct_time(","")
    string = string.replace(")","")
    string_array = string.split(",")
    new_date =  string_array[0].replace("tm_year","").replace("=","").strip() + "-"+ string_array[1].replace("tm_mon","").replace("=","").strip() + "-" + string_array[2].replace("tm_mday","").replace("=","").strip()
    return new_date
    

def science_daily_parser(): 
    feed_array = []  
    parsed_feeds = feedparser.parse("https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml")
    for each_entry in parsed_feeds['entries']:
        dict_ = {}
        try:
            dict_["feed_image"] = str(each_entry['media_thumbnail'][0]['url'])
            dict_["feed_link"] = str(each_entry['links'][0]['href'])
            struct_to_normal_date = struct_time_corrector(str(each_entry['published_parsed']))
            # print ("struct_to_normal_date : ",struct_to_normal_date )
            struct_to_normal_date = dateparser.parse(struct_to_normal_date, settings={'TIMEZONE': 'UTC'}).strftime("%Y-%m-%d")
            dict_["feed_published"] = struct_to_normal_date
            dict_["feed_title"] =  str(each_entry['title'])
            current_date_time = dateparser.parse(str(datetime.datetime.now()), settings={'TIMEZONE': 'UTC'}).strftime("%Y-%m-%d")
            delta = days_between(current_date_time , struct_to_normal_date)
            dict_["day_difference"] = delta
            dict_["week_difference"] = int(delta/7)
            # print (dict_)
            feed_array.append(dict_)
        except:
            ""
    return feed_array

feed_list  = ['https://www.artificial-intelligence.blog/news?format=rss','http://news.mit.edu/rss/topic/artificial-intelligence2']
master_dict = []
for each_feed in feed_list:
    parsed_feeds = feedparser.parse(each_feed)
    feed_arry = get_relevant_fields(parsed_feeds)
    master_dict.extend(feed_arry)

master_dict.extend(science_daily_parser())

master_html = ""
newlist = sorted(master_dict, key=lambda k: k['week_difference']) 


FEED_POST_TRMPLATE = ""
for some_sample in newlist:    
    if some_sample['week_difference'] == 0:
        feed_link = some_sample["feed_link"]
        feed_image = some_sample["feed_image"]
        feed_title = some_sample["feed_title"]
        feed_published = some_sample["feed_published"]
        FEED_POST_TRMPLATE += '<div class="chapter"><a href='+feed_link+'><img src='+feed_image+' alt='+feed_title+'></a><div class="chapter_inner"><p class="chapter_number"'+feed_published+'</p><a href='+feed_link+' target = "blank"><h3 class="chapter_title">'+feed_title+'</h3></a></div></div>'
sub_html = '<section class="chapters cf"><h1 class="sub_title" style="text-align: center; color: aliceblue;">This Week</h1><div class="wrapper flex-row">'+FEED_POST_TRMPLATE+'</div></section>'
master_html += sub_html
    
FEED_POST_TRMPLATE = ""
for some_sample in newlist:    
    if some_sample['week_difference'] == 1:
        feed_link = some_sample["feed_link"]
        feed_image = some_sample["feed_image"]
        feed_title = some_sample["feed_title"]
        feed_published = some_sample["feed_published"]
        FEED_POST_TRMPLATE += '<div class="chapter"><a href='+feed_link+'  target = "blank"><img src='+feed_image+' alt='+feed_title+'></a><div class="chapter_inner"><p class="chapter_number"'+feed_published+'</p><a href='+feed_link+' target = "blank"><h3 class="chapter_title">'+feed_title+'</h3></a></div></div>'
sub_html = '<section class="chapters cf"><h1 class="sub_title" style="text-align: center; color: aliceblue;">A Week Ago</h1><div class="wrapper flex-row">'+FEED_POST_TRMPLATE+'</div></section>'
master_html += sub_html
    
FEED_POST_TRMPLATE = ""
for some_sample in newlist:    
    if (some_sample['week_difference'] == 2) or (some_sample['week_difference'] == 3) :
        feed_link = some_sample["feed_link"]
        feed_image = some_sample["feed_image"]
        feed_title = some_sample["feed_title"]
        feed_published = some_sample["feed_published"]
        FEED_POST_TRMPLATE += '<div class="chapter"><a href='+feed_link+'  target = "blank"><img src='+feed_image+' alt='+feed_title+'></a><div class="chapter_inner"><p class="chapter_number"'+feed_published+'</p><a href='+feed_link+' target = "blank"><h3 class="chapter_title">'+feed_title+'</h3></a></div></div>'
sub_html = '<section class="chapters cf"><h1 class="sub_title" style="text-align: center; color: aliceblue;">Older</h1><div class="wrapper flex-row">'+FEED_POST_TRMPLATE+'</div></section>'
master_html += sub_html
    

title_string = '---\nlayout: default\ntitle: feeds\npermalink: /feeds/\n---\n\n'
template_text  = open("_pages/feed_template.html").read()
template_text = template_text.replace("!!! FEED_PAGE_TITLE !!!",title_string)    

template_text = template_text.replace("!!! FEED_PAGE_CARDS !!!",master_html)

output = open("_pages/feeds.html","w")
output.write(template_text)

