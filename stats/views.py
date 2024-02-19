from rest_framework import viewsets, status
from rest_framework.permissions import IsAuthenticated
from .serializers import QuotaCalculationSerializer
from .models import TokenUsage


class QuotaViewSet(viewsets.ModelViewSet):
    serializer_class = QuotaCalculationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # 先查询当前用户是否有TokenUsage记录，如果没有则创建一个
        if not TokenUsage.objects.filter(user=self.request.user).exists():
            TokenUsage.objects.create(user=self.request.user)
        result = TokenUsage.objects.get(user=self.request.user)
        return [result]
