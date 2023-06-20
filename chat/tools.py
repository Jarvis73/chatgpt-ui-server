from utils.search_prompt import compile_prompt
from utils.duckduckgo_search import web_search, SearchRequest


def _web_search(message, args):
    """search the web
    """
    search_results = web_search(SearchRequest(message, ua=args['ua']), num_results=5)
    message_content = compile_prompt(search_results, message, default_prompt=args['default_prompt'])
    return message_content


TOOL_LIST = {
    'web_search': _web_search,
}
