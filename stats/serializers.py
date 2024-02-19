from rest_framework import serializers
from .models import TokenUsage


class QuotaSerializer(serializers.ModelSerializer):
    class Meta:
        model = TokenUsage
        fields = ['tokens', 'balance', 'remain', 'paid_tokens', 'paid_balance', 'paid_remain']


class QuotaCalculationSerializer(QuotaSerializer):
    remain = serializers.SerializerMethodField()
    paid_remain = serializers.SerializerMethodField()

    def get_remain(self, obj):
        return int(obj.balance * 50_000)

    def get_paid_remain(self, obj):
        return int(obj.paid_balance * 50_000)
