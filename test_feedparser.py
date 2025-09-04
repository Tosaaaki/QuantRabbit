import feedparser

url = "http://feeds.bbci.co.uk/news/rss.xml"
parsed = feedparser.parse(url)

print(f"Number of entries found: {len(parsed.entries)}")

if parsed.entries:
    print("First entry title:", parsed.entries[0].title)
else:
    print("No entries found or parsing failed.")