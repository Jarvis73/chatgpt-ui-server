from django.contrib import admin
from rangefilter.filters import NumericRangeFilterBuilder

from .models import TokenUsage, Profile


@admin.register(TokenUsage)
class TokenUsageAdmin(admin.ModelAdmin):
    list_display = ('user', 'tokens')
    search_fields = ('user__username',)
    list_filter = (
        ('tokens', NumericRangeFilterBuilder()),
    )

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'vip')
    search_fields = ('user__username',)
