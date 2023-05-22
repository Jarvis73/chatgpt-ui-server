from django.urls import include, path
from rest_framework import routers
from .views import ConversationViewSet, MessageViewSet, PromptViewSet, MaskViewSet, SettingViewSet

router = routers.SimpleRouter()
router.register(r'conversations', ConversationViewSet, basename='conversationModel')
router.register(r'messages', MessageViewSet, basename='messageModel')
router.register(r'prompts', PromptViewSet, basename='promptModel')
router.register(r'masks', MaskViewSet, basename='maskModel')
router.register(r'settings', SettingViewSet, basename='settingModel')

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
]