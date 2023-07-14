import hashlib

from rest_framework.response import Response
from rest_framework import status
from dj_rest_auth.app_settings import api_settings
from dj_rest_auth.registration.views import RegisterView
from dj_rest_auth.utils import jwt_encode
from chat.models import Setting
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

        # 获取邀请码余量
        try:
            code_inventory_quantity = int(Setting.objects.get(name='code_inventory_quantity').value)
        except Setting.DoesNotExist:
            code_inventory_quantity = 0

        if not code_inventory_quantity:
            return Response({'detail': 'Registration is not yet open.'}, status=status.HTTP_403_FORBIDDEN)

        # 判断邀请码是否正确
        try:
            open_code = Setting.objects.get(name='open_code').value == 'True'
        except Setting.DoesNotExist:
            open_code = True
        if open_code is True:
            # 使用 sha256 来验证邀请码, 以防邀请码泄露
            if 'code' not in request.data or \
                    hashlib.sha256(request.data['code'].encode('utf-8')).hexdigest() \
                    not in ['065116489e21a5c20337b797c839e5c75c014924291ac33cb21989623e89adeb',
                            '2da96cc671ddb485ca378e39e259585a58c4c4f68555bdb0457d3c5553e9ec8b']:
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
            code_inventory_quantity = Setting.objects.get(name='code_inventory_quantity')
            if int(code_inventory_quantity.value) > 0:
                code_inventory_quantity.value = int(code_inventory_quantity.value) - 1
                code_inventory_quantity.save()
        except Setting.DoesNotExist:
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