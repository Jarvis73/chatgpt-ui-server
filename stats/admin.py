from django.contrib import admin
from .models import TokenUsage, Profile


@admin.register(TokenUsage)
class TokenUsageAdmin(admin.ModelAdmin):
    list_display = ('user', 'tokens', 'balance', 'paid_tokens', 'paid_balance')
    search_fields = ('user__username',)

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'vip')
    search_fields = ('user__username',)
