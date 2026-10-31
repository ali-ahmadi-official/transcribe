import json
import os
import re
import threading
import time
import uuid
from html import escape
from html.parser import HTMLParser
from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.template.defaultfilters import linebreaks_filter
from django.urls import reverse
from django.views.decorators.http import require_POST

from .models import Transcription
from .whisper_loader import get_model


JOBS = {}
JOBS_LOCK = threading.Lock()
TRANSCRIBE_LOCK = threading.Lock()
TMP_DIR = Path(settings.MEDIA_ROOT) / 'tmp_uploads'
ALLOWED_EXT = {'.mp3', '.wav', '.m4a', '.ogg', '.oga', '.opus', '.aac',
               '.flac', '.webm', '.mp4', '.wma', '.amr'}


ALLOWED_TAGS = {'p', 'br', 'b', 'strong', 'i', 'em', 'u', 'ul', 'ol', 'li',
                'h1', 'h2', 'h3', 'div', 'span', 'blockquote'}
ALIGN_RE = re.compile(r'text-align\s*:\s*(right|left|center|justify)', re.I)
LANGS = {
    'fa': {
        'dir': 'rtl',
        'prompt': (
            "گزارش رادیولوژی MRI و CT با تزریق IV Contrast. "
            "سکانس‌های T1 و T2 و FLAIR و STIR و GRE، "
            "تصاویر DWI و نقشه ADC، مطالعه Diffusion و Perfusion، "
            "تصاویر Post Contrast، MRA و MRV، و بازسازی‌های MIP و MPR."
        ),
    },
    'en': {
        'dir': 'ltr',
        'prompt': (
            "Radiology report for MRI and CT with IV contrast. "
            "T1, T2, FLAIR, STIR and GRE sequences, DWI images and ADC map, "
            "diffusion and perfusion studies, post-contrast images, "
            "MRA, MRV, MIP and MPR reconstructions."
        ),
    },
}
DEFAULT_LANG = 'fa'


class _Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in ALLOWED_TAGS:
            return
        if tag == 'br':
            self.out.append('<br>')
            return
        style = ''
        dir_attr = ''
        for k, v in attrs:
            if k == 'style' and v:
                m = ALIGN_RE.search(v)
                if m:
                    style = f' style="text-align: {m.group(1).lower()}"'
            elif k == 'dir' and v in ('rtl', 'ltr'):
                dir_attr = f' dir="{v}"'
        self.out.append(f'<{tag}{dir_attr}{style}>')
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.stack:
            while self.stack:
                t = self.stack.pop()
                self.out.append(f'</{t}>')
                if t == tag:
                    break

    def handle_data(self, data):
        self.out.append(escape(data))

    def result(self):
        while self.stack:
            self.out.append(f'</{self.stack.pop()}>')
        return ''.join(self.out)


def sanitize_html(raw):
    s = _Sanitizer()
    s.feed(raw or '')
    s.close()
    return s.result()


# ------------------------- helpers -----------------------------------
def _update_job(job_id, **kw):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kw)


def _remove_tmp(job):
    try:
        os.remove(job['path'])
    except OSError:
        pass


def _cleanup_old(max_age=6 * 3600):
    now = time.time()
    with JOBS_LOCK:
        old = [k for k, v in JOBS.items()
               if now - v['created'] > max_age and v['status'] in ('done', 'failed')]
        for k in old:
            _remove_tmp(JOBS.pop(k))


def _process(job_id):
    try:
        with TRANSCRIBE_LOCK:
            _update_job(job_id, status='processing')
            with JOBS_LOCK:
                path = JOBS[job_id]['path']
                lang = JOBS[job_id]['language']
            model = get_model()

            segments, info = model.transcribe(
                path,
                language=lang,
                beam_size=5,
                vad_filter=True,
                initial_prompt=LANGS[lang]['prompt'],
                condition_on_previous_text=False,
            )
            text = ' '.join(seg.text.strip() for seg in segments).strip()
        _update_job(job_id, status='done', text=text)
    except Exception as e:
        _update_job(job_id, status='failed', error=str(e))


# ------------------------- views -------------------------------------
def index(request):
    items = Transcription.objects.filter(status='done').order_by('-created_at')
    return render(request, 'index.html', {'items': items})


@require_POST
def upload_audio(request):
    audio = request.FILES.get('audio')
    if not audio:
        return JsonResponse({'error': 'فایلی ارسال نشد'}, status=400)
    ext = Path(audio.name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return JsonResponse({'error': 'فرمت فایل پشتیبانی نمی‌شود'}, status=400)

    lang = request.POST.get('language', DEFAULT_LANG)
    if lang not in LANGS:
        lang = DEFAULT_LANG

    _cleanup_old()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    path = TMP_DIR / f'{job_id}{ext}'
    with open(path, 'wb') as f:
        for chunk in audio.chunks():
            f.write(chunk)

    with JOBS_LOCK:
        JOBS[job_id] = {
            'status': 'pending', 'text': '', 'error': '',
            'path': str(path), 'filename': audio.name, 'created': time.time(),
            'language': lang,
        }
    threading.Thread(target=_process, args=(job_id,), daemon=True).start()
    return JsonResponse({'id': job_id, 'status': 'pending'})


def status(request, job_id):
    with JOBS_LOCK:
        job = JOBS.get(str(job_id))
        data = dict(job) if job else None
    if not data:
        return JsonResponse({'status': 'failed', 'error': 'این تبدیل پیدا نشد (ممکن است سرور ری‌استارت شده باشد)'}, status=404)
    return JsonResponse({'status': data['status'], 'error': data['error']})


def result(request, job_id):
    with JOBS_LOCK:
        job = JOBS.get(str(job_id))
        job = dict(job) if job else None
    if not job or job['status'] != 'done':
        return redirect('index')
    return render(request, 'result.html', {
        'mode': 'job',
        'title': Path(job['filename']).stem,
        'content_html': linebreaks_filter(job['text'], autoescape=True),
        'save_url': reverse('save_result', args=[job_id]),
        'discard_url': reverse('discard_result', args=[job_id]),
        'audio_url': '',
        'lang': job['language'],
        'dir': LANGS[job['language']]['dir'],
    })


@require_POST
def save_result(request, job_id):
    key = str(job_id)
    with JOBS_LOCK:
        job = JOBS.get(key)
        job = dict(job) if job else None
    if not job or job['status'] != 'done':
        return JsonResponse({'error': 'این نتیجه دیگر در دسترس نیست'}, status=404)
    try:
        html = json.loads(request.body).get('html', '')
    except (ValueError, AttributeError):
        return JsonResponse({'error': 'درخواست نامعتبر'}, status=400)

    obj = Transcription(text=sanitize_html(html), status='done', language=job['language'])
    with open(job['path'], 'rb') as f:
        obj.audio_file.save(job['filename'], File(f), save=False)
    obj.save()

    with JOBS_LOCK:
        JOBS.pop(key, None)
    _remove_tmp(job)
    return JsonResponse({'url': reverse('detail', args=[obj.id])})


@require_POST
def discard_result(request, job_id):
    with JOBS_LOCK:
        job = JOBS.pop(str(job_id), None)
    if job:
        _remove_tmp(job)
    return JsonResponse({'url': reverse('index')})


def detail(request, pk):
    obj = get_object_or_404(Transcription, pk=pk)
    return render(request, 'result.html', {
        'mode': 'saved',
        'title': Path(obj.audio_file.name).stem,
        'content_html': obj.text,
        'save_url': reverse('update_saved', args=[pk]),
        'discard_url': '',
        'audio_url': obj.audio_file.url,
        'lang': obj.language,
        'dir': LANGS.get(obj.language, LANGS['fa'])['dir'],
    })


@require_POST
def update_saved(request, pk):
    obj = get_object_or_404(Transcription, pk=pk)
    try:
        html = json.loads(request.body).get('html', '')
    except (ValueError, AttributeError):
        return JsonResponse({'error': 'درخواست نامعتبر'}, status=400)
    obj.text = sanitize_html(html)
    obj.save(update_fields=['text'])
    return JsonResponse({'ok': True})
