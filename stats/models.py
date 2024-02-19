from django.db import models
from django.contrib.auth.models import User


class TokenUsage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    tokens = models.IntegerField(default=0)             # freeUsage
    balance = models.FloatField(default=2000.0)        # free balance
    paid_tokens = models.IntegerField(default=0)        # paidUsage
    paid_balance = models.FloatField(default=0.0)       # paid balance
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class Profile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, unique=True)
    vip = models.BooleanField(default=False)
