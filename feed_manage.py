
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

        FEED_POST_TRMPLATE = '<div class="gallery"><a target="_blank" href="'+feed_link+'"><img src="'+feed_image+'" alt="Trolltunga Norway" width="300" height="200"></a><div class="desc">'+feed_title+'</div></div></div>'
        card_string = card_string + FEED_POST_TRMPLATE
        
        # print("#################################################")

    except:
        ""  

template_text = template_text.replace("!!! FEED_PAGE_CARDS !!!",card_string)

output = open("_pages/feeds.html","w")
output.write(template_text)

