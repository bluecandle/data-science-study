# what are spiders?

일단 뭔가 웹상에서 돌아다닌다는 의미로 쓰이는 것 같긴 한데, doc 을 읽어보자.
<a href ="https://docs.scrapy.org/en/latest/topics/spiders.html#topics-spiders">doc</a>
Spiders are classes which define how a certain site (or a group of sites) will be scraped, including how to perform the crawl (i.e. follow links) and how to extract structured data from their pages (i.e. scraping items). In other words, Spiders are the place where you define the custom behaviour for crawling and parsing pages for a particular site (or, in some cases, a group of sites)

# attributes 
name: identifies the Spider. It must be unique within a project, that is, you can’t set the same name for different Spiders.

start_requests(): must return an iterable of Requests (you can return a list of requests or write a generator function) which the Spider will begin to crawl from. Subsequent requests will be generated successively from these initial requests.

parse(): a method that will be called to handle the response downloaded for each of the requests made. The response parameter is an instance of TextResponse that holds the page content and has further helpful methods to handle it.

# how it works ( request , response objects )
Scrapy schedules the scrapy.Request objects returned by the start_requests method of the Spider. Upon receiving a response for each one, it instantiates Response objects and calls the callback method associated with the request (in this case, the parse method) passing the response as argument.

# 


