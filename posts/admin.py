from django.contrib import admin

# Register your models here.
from .models import Post, Stocks, Input, Report, Candle, newsletter

admin.site.register(Post)
admin.site.register(Stocks)
admin.site.register(Input)
admin.site.register(Report)
admin.site.register(Candle)
admin.site.register(newsletter)