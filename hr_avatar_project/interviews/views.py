# hr_avatar_project/views.py
from django.shortcuts import render

def home_page(request):
    """Главная страница сайта"""
    return render(request, 'interviews/index.html')

# interviews/views.py
from django.shortcuts import render

# def interview_page(request, candidate_id=None):
#     """Страница с WebSocket интерфейсом"""
#     context = {
#         'candidate_id': candidate_id or 'default',
#         'websocket_url': f'ws://{request.get_host()}/ws/index/{candidate_id or "default"}/'
#     }
#     return render(request, 'interviews/index.html', context)