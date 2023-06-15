import os
import json
import openai
import time
import datetime
import tiktoken

from provider.models import ApiKey
from stats.models import TokenUsage, Profile
from .models import Conversation, Message, Setting, Prompt, Mask
from django.conf import settings
from django.http import StreamingHttpResponse
from django.forms.models import model_to_dict
from django.core.cache import cache
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes, action
from .serializers import ConversationSerializer, MessageSerializer, PromptSerializer, MaskSerializer, SettingSerializer
from utils.search_prompt import compile_prompt
from utils.duckduckgo_search import web_search, SearchRequest


class SettingViewSet(viewsets.ModelViewSet):
    serializer_class = SettingSerializer
    # permission_classes = [IsAuthenticated]

    def get_queryset(self):
        available_names = [
            'open_registration',
            'open_web_search',
            'open_api_key_setting',
            'open_frugal_mode_control',
            'open_code',
            'code_inventory_quantity',
            'azure_api_base',
        ]
        return Setting.objects.filter(name__in=available_names)

    def http_method_not_allowed(self, request, *args, **kwargs):
        if request.method != 'GET':
            return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)
        return super().http_method_not_allowed(request, *args, **kwargs)


class ConversationViewSet(viewsets.ModelViewSet):
    serializer_class = ConversationSerializer
    # authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user).order_by('-created_at')

    @action(detail=False, methods=['delete'])
    def delete_all(self, request):
        queryset = self.filter_queryset(self.get_queryset())
        queryset.delete()
        return Response(status=204)


class MessageViewSet(viewsets.ModelViewSet):
    serializer_class = MessageSerializer
    # authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Message.objects.all()

    def get_queryset(self):
        queryset = super().get_queryset()
        conversationId = self.request.query_params.get('conversationId')
        if conversationId:
            queryset = queryset.filter(conversation__user=self.request.user) \
                .filter(conversation_id=conversationId).order_by('created_at')
            return queryset
        return queryset


class PromptViewSet(viewsets.ModelViewSet):
    serializer_class = PromptSerializer
    # authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Prompt.objects.filter(user=self.request.user).order_by('-created_at')

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        serializer.validated_data['user'] = request.user

        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    @action(detail=False, methods=['delete'])
    def delete_all(self, request):
        queryset = self.filter_queryset(self.get_queryset())
        queryset.delete()
        return Response(status=204)


class MaskViewSet(viewsets.ModelViewSet):
    serializer_class = MaskSerializer
    # authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Mask.objects.filter(user=self.request.user).order_by('-created_at')

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        serializer.validated_data['user'] = request.user

        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    @action(detail=False, methods=['delete'])
    def delete_all(self, request):
        queryset = self.filter_queryset(self.get_queryset())
        queryset.delete()
        return Response(status=204)


MODELS = {
    # 'gpt-3.5-turbo': {
    #     'name': 'gpt-3.5-turbo-0301',
    #     'key_name': 'gpt-3.5-turbo-azure',
    #     'max_tokens': 4096,
    #     'max_prompt_tokens': 3096,
    #     'max_response_tokens': 1000,
    #     'azure': True,
    #     'kwargs': {
    #         'deployment_id': 'gpt35'
    #     },
    # },
    'gpt-3.5-turbo': {
        'name': 'gpt-3.5-turbo-0613',
        'key_name': 'gpt-3.5-turbo',
        'max_tokens': 4096,
        'max_prompt_tokens': 3096,
        'max_response_tokens': 1000,
        'azure': False,
        'kwargs': {},
    },
    'gpt-3.5-turbo-16k': {
        'name': 'gpt-3.5-turbo-16k-0613',
        'key_name': 'gpt-3.5-turbo',
        'max_tokens': 16384,
        'max_prompt_tokens': 12384,
        'max_response_tokens': 4000,
        'azure': False,
        'kwargs': {},
    },
    'gpt-4': {
        'name': 'gpt-4-0613',
        'key_name': 'gpt-4',
        'max_tokens': 8192,
        'max_prompt_tokens': 6196,
        'max_response_tokens': 2000,
        'azure': False,
        'kwargs': {},
    }
}

MODEL_SET = {
    '3.5': {'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613'},
    '4': {'gpt-4-0314', 'gpt-4-0613'}
}


def sse_pack(event, data):
    # Format data as an SSE message
    packet = "event: %s\n" % event
    packet += "data: %s\n" % json.dumps(data)
    packet += "\n"
    return packet


@api_view(['POST'])
# @authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def gen_title(request):
    conversation_id = request.data.get('conversationId')
    prompt = request.data.get('prompt')
    conversation_obj = Conversation.objects.get(id=conversation_id)
    message = Message.objects.filter(conversation_id=conversation_id).order_by('created_at').first()
    openai_api_key = request.data.get('openaiApiKey') or None
    api_key = None

    model = MODELS['gpt-3.5-turbo']

    # if openai_api_key is None:
    #     openai_api_key = get_api_key_from_setting()

    if openai_api_key is None:
        api_key = get_api_key(request.user, model['key_name'])
        if api_key:
            openai_api_key = api_key.key
        else:
            return Response(
                {
                    'error': 'There is no available API key'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

    if prompt is None:
        prompt = 'Generate a short title for the following content, no more than 10 words.'

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message.message},
    ]

    my_openai = get_openai(model, openai_api_key)
    try:
        openai_response = my_openai.ChatCompletion.create(
            model=model['name'],
            messages=messages,
            max_tokens=256,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            **model['kwargs'],
        )
        completion_text = openai_response['choices'][0]['message']['content']
        title = completion_text.strip().replace('"', '')

        # increment the token count
        increase_token_usage(request.user, openai_response['usage']['total_tokens'], api_key)
    except Exception as e:
        print(e)
        title = 'Untitled Conversation'
    # update the conversation title
    conversation_obj.topic = title
    conversation_obj.save()

    return Response({
        'title': title
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def stop_conversation(request):
    # Stop current running conversation
    cache.set(request.user, 0, 300)
    return Response({})


@api_view(['POST'])
# @authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def conversation(request):
    # Set an is_running flag for responding the stop request
    cache.set(request.user, 1, 300)

    model_name = request.data.get('name')
    message = request.data.get('message')
    conversation_id = request.data.get('conversationId')
    request_max_response_tokens = request.data.get('max_tokens')
    temperature = request.data.get('temperature', 0.7)
    top_p = request.data.get('top_p', 1)
    frequency_penalty = request.data.get('frequency_penalty', 0)
    presence_penalty = request.data.get('presence_penalty', 0)
    web_search_params = request.data.get('web_search')
    openai_api_key = request.data.get('openaiApiKey') or None
    frugal_mode = request.data.get('frugalMode', False)
    mask_title = request.data.get('maskTitle', '')
    mask_avatar = request.data.get('maskAvatar', '')
    few_shot_messages = request.data.get('fewShotMask', [])
    api_key = None

    model = get_current_model(model_name, request_max_response_tokens)

    # if openai_api_key is None:
    #     openai_api_key = get_api_key_from_setting()

    if openai_api_key is None:
        api_key = get_api_key(request.user, model['key_name'])
        if api_key:
            openai_api_key = api_key.key
        else:
            return Response(
                {
                    'error': 'There is no available API key'
                },
                status=status.HTTP_403_FORBIDDEN
            )

    try:
        check_few_shot_messages(few_shot_messages)
        messages = build_messages(model, conversation_id, message, few_shot_messages, web_search_params, frugal_mode)

        if settings.DEBUG:
            print('messages:', messages)
    except Exception as e:
        print(e)
        return Response(
            {
                'error': e
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    def stream_content():
        my_openai = get_openai(model, openai_api_key)
        try:
            openai_response = my_openai.ChatCompletion.create(
                model=model['name'],
                messages=messages['messages'],
                max_tokens=model['max_response_tokens'],
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True,
                **model['kwargs'],
            )
        except Exception as e:
            yield sse_pack('error', {
                'error': 'Internal Server Error!'
            })
            print('openai error', e)
            return

        if conversation_id:
            # get the conversation
            conversation_obj = Conversation.objects.get(id=conversation_id)
            # update mask information
            modified = False
            if conversation_obj.mask_title != mask_title:
                conversation_obj.mask_title = mask_title
                modified = True
            if conversation_obj.mask_avatar != mask_avatar:
                conversation_obj.mask_avatar = mask_avatar
                modified = True
            str_few_shot_messages = json.dumps(few_shot_messages)
            if conversation_obj.mask != str_few_shot_messages:
                conversation_obj.mask = str_few_shot_messages
                modified = True
            if modified:
                conversation_obj.save()
        else:
            # create a new conversation
            conversation_obj = Conversation(user=request.user,
                                            mask_title=mask_title,
                                            mask_avatar=mask_avatar,
                                            mask=json.dumps(few_shot_messages))
            conversation_obj.save()

        # insert a new message
        message_obj = create_message(
            user=request.user,
            conversation_id=conversation_obj.id,
            message=message,
            messages=messages['messages'],
            tokens=messages['tokens'],
            api_key=api_key
        )
        yield sse_pack('userMessageId', {
            'userMessageId': message_obj.id,
        })

        collected_events = []
        completion_text = ''
        # iterate through the stream of events
        for idx, event in enumerate(openai_response):
            collected_events.append(event)  # save the event response
            if event['choices'][0]['finish_reason'] is not None:
                break
            if 'content' in event['choices'][0]['delta']:
                event_text = event['choices'][0]['delta']['content']
                completion_text += event_text  # append the text
                yield sse_pack('message', {'content': event_text})
                if model['azure']:
                    # We slow down the message output when using Azure
                    # because it is too fast!!!
                    time.sleep(0.02)
            # Check is_running every 10 ticks.
            if idx % 10 == 0 and cache.get(request.user) == 0:
                break

        ai_message_token = num_tokens_from_text(completion_text, model['name'])
        ai_message_obj = create_message(
            user=request.user,
            conversation_id=conversation_obj.id,
            message=completion_text,
            is_bot=True,
            tokens=ai_message_token,
            api_key=api_key
        )
        yield sse_pack('done', {
            'messageId': ai_message_obj.id,
            'conversationId': conversation_obj.id
        })

    response = StreamingHttpResponse(
        stream_content(),
        content_type='text/event-stream'
    )
    response['X-Accel-Buffering'] = 'no'
    response['Cache-Control'] = 'no-cache'
    return response


def check_few_shot_messages(messages):
    if not isinstance(messages, list):
        raise TypeError(f'`messages` must be a list, got {type(messages)}.')
    for item in messages:
        if not isinstance(item, dict):
            raise TypeError(f'items in `messages` must be dictionary, got {type(item)}')
        if 'role' not in item:
            raise ValueError(f'items in `messages` must contains a key "role"')
        if 'content' not in item:
            raise ValueError(f'items in `messages` must contains a key "content"')
        if len(item.keys()) > 2:
            raise ValueError(f'items in `messages` must only contains two keys "role" and "content", '
                             f'got {list(item.keys())}')

def create_message(user, conversation_id, message, is_bot=False, messages='', tokens=0, api_key=None):
    message_obj = Message(
        conversation_id=conversation_id,
        message=message,
        is_bot=is_bot,
        messages=messages,
        tokens=tokens
    )
    message_obj.save()

    increase_token_usage(user, tokens, api_key)

    return message_obj


def increase_token_usage(user, tokens, api_key=None):
    token_usage, created = TokenUsage.objects.get_or_create(user=user)
    token_usage.tokens += tokens
    token_usage.save()

    if api_key:
        api_key.token_used += tokens
        api_key.save()


def build_messages(
        model,
        conversation_id,
        new_message_content,
        few_shot_messages,
        web_search_params,
        frugal_mode=False):
    if conversation_id:
        ordered_messages = Message.objects.filter(conversation_id=conversation_id).order_by('created_at')
        ordered_messages_list = list(ordered_messages)
    else:
        ordered_messages_list = []

    ordered_messages_list.append({'is_bot': False, 'message': new_message_content})

    if frugal_mode:
        ordered_messages_list = ordered_messages_list[-1:]

    # Create preset messages
    system_messages = few_shot_messages or [{"role": "system", "content": "You are a helpful assistant."}]

    current_token_count = num_tokens_from_messages(system_messages, model['name'])

    max_token_count = model['max_prompt_tokens']

    messages = []

    result = {
        'messages': messages,
        'tokens': 0
    }

    while current_token_count < max_token_count and len(ordered_messages_list) > 0:
        message = ordered_messages_list.pop()
        if isinstance(message, Message):
            message = model_to_dict(message)
        role = "assistant" if message['is_bot'] else "user"
        if web_search_params is not None and len(messages) == 0:
            search_results = web_search(SearchRequest(message['message'], ua=web_search_params['ua']), num_results=5)
            message_content = compile_prompt(search_results, message['message'], default_prompt=web_search_params['default_prompt'])
        else:
            message_content = message['message']
        new_message = {"role": role, "content": message_content}
        new_token_count = num_tokens_from_messages(system_messages + messages + [new_message], model['name'])
        if new_token_count > max_token_count:
            if len(messages) > 0:
                break
            raise ValueError(
                f"Prompt is too long. Max token count is {max_token_count}, but prompt is {new_token_count} tokens long.")
        messages.insert(0, new_message)
        current_token_count = new_token_count

    result['messages'] = system_messages + messages
    result['tokens'] = current_token_count

    return result


def get_current_model(model_name, request_max_response_tokens):
    if model_name is None:
        model_name ="gpt-3.5-turbo"
    model = MODELS[model_name]
    if request_max_response_tokens is not None:
        model['max_response_tokens'] = int(request_max_response_tokens)
        model['max_prompt_tokens'] = model['max_tokens'] - model['max_response_tokens']
    return model


def get_api_key_from_setting():
    row = Setting.objects.filter(name='openai_api_key').first()
    if row and row.value != '':
        return row.value
    return None


def get_api_key(user, key_name='gpt-3.5-turbo'):
    try:
        api_key = ApiKey.objects.filter(
            is_enabled=True, remark__iexact=key_name).order_by('token_used').first()
        # gpt-4 is only for vip
        if api_key is not None and api_key.remark.lower() == 'gpt-4':
            user = Profile.objects.filter(user=user)
            if not user.exists() or not user.first().vip:
                api_key = None
    except ApiKey.DoesNotExist as error:
        api_key = None

    return api_key

def num_tokens_from_text(text, model="gpt-3.5-turbo-0301"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_text(text, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_text(text, model="gpt-4-0314")

    if model in MODEL_SET['3.5'] or MODEL_SET['4']:
        return len(encoding.encode(text))

    raise NotImplementedError(f"""num_tokens_from_text() is not implemented for model {model}.""")


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model in MODEL_SET['3.5']:
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model in MODEL_SET['4']:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_openai(model, openai_api_key):
    if model['azure']:
        openai.api_type = "azure"
        openai.api_key = openai_api_key
        api_base = Setting.objects.filter(name='azure_api_base').first()
        if not api_base:
            raise ValueError('Missing azure_api_base.')
        openai.api_base = api_base.value
        openai.api_version = "2023-03-15-preview"
    else:
        openai.api_type = "open_ai"
        openai.api_key = openai_api_key
        openai.api_base = 'https://api.openai.com/v1'
        openai.api_version = None
        proxy = os.getenv('OPENAI_API_PROXY')
        if proxy:
            openai.api_base = proxy
    return openai
