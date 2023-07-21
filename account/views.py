import hashlib

from rest_framework.response import Response
from rest_framework import status
from dj_rest_auth.app_settings import api_settings
from dj_rest_auth.registration.views import RegisterView
from dj_rest_auth.utils import jwt_encode
from chat.models import Setting
from provider.models import InvCode
from allauth.account import signals, app_settings as allauth_account_settings
from allauth.account.utils import perform_login
from rest_framework.exceptions import ValidationError


class RegistrationView(RegisterView):
    def create(self, request, *args, **kwargs):
        # 判断是否开启注册功能
        try:
            open_registration = Setting.objects.get(name='open_registration').value == 'True'
        except Setting.DoesNotExist:
            open_registration = True

        if open_registration is False:
            return Response({'detail': 'Registration is not yet open.'}, status=status.HTTP_403_FORBIDDEN)

        # 判断邀请码是否正确
        try:
            open_code = Setting.objects.get(name='open_code').value == 'True'
        except Setting.DoesNotExist:
            open_code = True
        if open_code is True:
            inv_code = request.data.get('code')
            try:
                inv_code_obj = InvCode.objects.get(code=inv_code, available_uses__gt=0)
            except InvCode.DoesNotExist:
                return Response({'detail': 'Invalid invitation code.'}, status=status.HTTP_403_FORBIDDEN)
        # 执行注册
        serializer = self.get_serializer(data=request.data)
        try:
            serializer.is_valid(raise_exception=True)
            user = self.perform_create(serializer)
        except ValidationError as error:
            return Response({'detail': list(error.detail.values())[0][0]}, status=status.HTTP_400_BAD_REQUEST)

        headers = self.get_success_headers(serializer.data)
        data = self.get_response_data(user)

        data['email_verification_required'] = allauth_account_settings.EMAIL_VERIFICATION

        if data:
            response = Response(
                data,
                status=status.HTTP_201_CREATED,
                headers=headers,
            )
        else:
            response = Response(status=status.HTTP_204_NO_CONTENT, headers=headers)

        return response

    def perform_create(self, serializer):
        user = serializer.save(self.request)

        # 注册成功, 减少邀请码数量
        try:
            inv_code_obj = InvCode.objects.get(code=self.request.data.get('code'), available_uses__gt=0)
            inv_code_obj.available_uses -= 1
            inv_code_obj.save()
        except InvCode.DoesNotExist:
            pass

        if allauth_account_settings.EMAIL_VERIFICATION != \
                allauth_account_settings.EmailVerificationMethod.MANDATORY:
            if api_settings.USE_JWT:
                self.access_token, self.refresh_token = jwt_encode(user)
            elif not api_settings.SESSION_LOGIN:
                # Session authentication isn't active either, so this has to be
                #  token authentication
                api_settings.TOKEN_CREATOR(self.token_model, user, serializer)

        complete_signup(
            self.request._request, user,
            allauth_account_settings.EMAIL_VERIFICATION,
            None,
        )
        return user


def complete_signup(request, user, email_verification, success_url, signal_kwargs=None):
    if signal_kwargs is None:
        signal_kwargs = {}
    signals.user_signed_up.send(
        sender=user.__class__, request=request, user=user, **signal_kwargs
    )
    return perform_login(
        request,
        user,
        email_verification=email_verification,
        signup=False,
        redirect_url=success_url,
        signal_kwargs=signal_kwargs,
    )
