
import feedparser
import json
d = feedparser.parse('https://www.artificial-intelligence.blog/news?format=rss')


title_string = '---\nlayout: default\ntitle: feeds\npermalink: /feeds/\n---\n\n'
template_text  = open("_pages/feed_template.html").read()
print(template_text)
template_text = template_text.replace("!!! FEED_PAGE_TITLE !!!",title_string)    


card_string = ""
for each_entry_no in range(len(d.entries)):
    try:
        # print("#################################################")
        feed_image = json.dumps(d.entries[each_entry_no]['media_content'][0]['url'])
        # print(feed_image)
        feed_link = json.dumps(d.entries[each_entry_no]['link'])
        # print(feed_link)
        feed_title = json.dumps(d.entries[each_entry_no]['title'])
        # print(feed_title)
        feed_published = json.dumps(d.entries[each_entry_no]['published'])
        # print(feed_published)

        FEED_POST_TRMPLATE = '<div class="chapter"><a href='+feed_link+'><img src='+feed_image+' alt='+feed_title+'></a><div class="chapter_inner"><p class="chapter_number"'+feed_published+'</p><a href='+feed_link+'s><h3 class="chapter_title">'+feed_title+'</h3></a></div></div>'
        card_string = card_string + FEED_POST_TRMPLATE
        
        # print("#################################################")

    except:
        ""  

template_text = template_text.replace("!!! FEED_PAGE_CARDS !!!",card_string)

output = open("_pages/feeds.html","w")
output.write(template_text)

