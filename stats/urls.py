from django.urls import include, path
from rest_framework import routers
from .views import QuotaViewSet

router = routers.SimpleRouter()
router.register(r'quota', QuotaViewSet, basename='quotaModel')

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
]