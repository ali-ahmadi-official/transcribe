import json
import mimetypes
import os
import re
import threading
import time
import uuid
from functools import wraps
from html import escape
from html.parser import HTMLParser
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.files import File
from django.core.exceptions import PermissionDenied, ValidationError
from django.db.models import Q
from django.db.models import Count
from django.http import Http404, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.template.defaultfilters import linebreaks_filter
from django.urls import reverse
from django.views.decorators.http import require_POST

from .models import Transcription
from .whisper_loader import get_model
from .llm_loader import get_llm

User = get_user_model()

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

SYSTEM_PROMPT = """You are a medical reporting assistant for physicians and radiologists.
You receive a text that is either the doctor's dictated findings, or a short request
(for example "ultrasound report, fatty liver grade 1"), optionally followed by extra
instructions from the doctor. Write a complete, professional medical report.

Rules:
- Write in {language_name}. Keep standard medical terms (MRI, CT, T1, T2, FLAIR, DWI, ADC, IV Contrast, ...) in English.
- Use only findings that the doctor stated or that are directly implied by the requested report type.
  Never invent measurements, sizes, dates or patient data. If a needed value is missing, write "____".
- Structure: title (h2), Technique, Findings (bullet list), Impression (numbered list).
- Output ONLY HTML using these tags: h2, h3, p, ul, ol, li, strong, em, br.
  No markdown, no code fences, no text before or after the report.
- Formal and concise tone."""

LANG_NAMES = {'fa': 'Persian (Farsi)', 'en': 'English'}


# ------------------------- sanitizer ---------------------------------
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


# ------------------------- auth helpers ------------------------------
def is_admin(user):
    return user.is_authenticated and (user.is_staff or user.is_superuser)


def display_name(user):
    if not user:
        return '—'
    return user.get_full_name().strip() or user.username


def api_login_required(view):
    """برای endpointهای JSON: به‌جای ریدایرکت به صفحه لاگین، 401 JSON برمی‌گرداند."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'نشست شما منقضی شده؛ دوباره وارد شوید'}, status=401)
        return view(request, *args, **kwargs)
    return wrapper


def _clean_patient(post):
    name = re.sub(r'\s+', ' ', post.get('patient_name') or '').strip()
    nid = post.get('patient_national_id')
    if not name:
        return None, None, 'نام بیمار را وارد کنید'
    if len(name) > 150:
        return None, None, 'نام بیمار خیلی طولانی است'
    return name, nid, None


def _doctors_qs(exclude_user=None):
    qs = User.objects.filter(is_active=True, is_staff=False, is_superuser=False)
    if exclude_user is not None:
        qs = qs.exclude(pk=exclude_user.pk)
    return qs.order_by('first_name', 'last_name', 'username')


# ------------------------- job helpers -------------------------------
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


def _get_job(request, job_id):
    """کپی job را فقط به صاحبش (کسی که آپلود کرده) می‌دهد."""
    with JOBS_LOCK:
        job = JOBS.get(str(job_id))
        job = dict(job) if job else None
    if not job or job.get('owner_id') != request.user.id:
        return None
    return job


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


def _generate(job_id, text, doctor_prompt):
    try:
        with TRANSCRIBE_LOCK:
            with JOBS_LOCK:
                lang = JOBS[job_id]['language']
            html = generate_report(text, lang, doctor_prompt)
        if '<' not in html:
            html = linebreaks_filter(html, autoescape=True)
        _update_job(job_id, ai_status='done', report_html=sanitize_html(html))
    except Exception as e:
        _update_job(job_id, ai_status='failed', ai_error=str(e))


def generate_report(text, language, doctor_prompt=''):
    system = SYSTEM_PROMPT.format(language_name=LANG_NAMES.get(language, 'Persian (Farsi)'))
    user = f"Text:\n{text}"
    if doctor_prompt.strip():
        user += f"\n\nDoctor's extra instructions:\n{doctor_prompt.strip()}"

    out = get_llm().create_chat_completion(
        messages=[{'role': 'system', 'content': system},
                  {'role': 'user', 'content': user}],
        temperature=0.2,
        max_tokens=1500,
    )
    html = out['choices'][0]['message']['content'].strip()
    return re.sub(r'^```(?:html)?\s*|\s*```$', '', html)


# ------------------------- protected audio ---------------------------
def _iter_range(f, length, chunk=64 * 1024):
    try:
        while length > 0:
            data = f.read(min(chunk, length))
            if not data:
                break
            length -= len(data)
            yield data
    finally:
        f.close()


def _serve_file(request, path, content_type):
    """ارسال فایل با پشتیبانی از Range (برای جلو/عقب کردن پلیر)."""
    size = os.path.getsize(path)
    start, end, status = 0, size - 1, 200
    m = re.match(r'bytes=(\d*)-(\d*)$', request.headers.get('Range', '').strip())
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            start = int(m.group(1))
            if m.group(2):
                end = min(int(m.group(2)), size - 1)
        else:
            start = max(size - int(m.group(2)), 0)
        if start > end or start >= size:
            resp = HttpResponse(status=416)
            resp['Content-Range'] = f'bytes */{size}'
            return resp
        status = 206
    length = max(end - start + 1, 0)
    f = open(path, 'rb')
    f.seek(start)
    resp = StreamingHttpResponse(_iter_range(f, length), status=status, content_type=content_type)
    resp['Content-Length'] = str(length)
    resp['Accept-Ranges'] = 'bytes'
    resp['Cache-Control'] = 'private, no-store'
    if status == 206:
        resp['Content-Range'] = f'bytes {start}-{end}/{size}'
    return resp


@login_required
def audio(request, pk):
    obj = get_object_or_404(Transcription.objects.visible_to(request.user), pk=pk)
    try:
        path = obj.audio_file.path
    except (ValueError, NotImplementedError):
        raise Http404
    if not os.path.exists(path):
        raise Http404
    return _serve_file(request, path, mimetypes.guess_type(path)[0] or 'application/octet-stream')


# ------------------------- views -------------------------------------
@login_required
def index(request):
    q = (request.GET.get('q') or '').strip()
    items = (Transcription.objects.visible_to(request.user)
             .filter(status='done').select_related('owner'))
    if q:
        items = items.filter(Q(patient_name__icontains=q) |
                             Q(patient_national_id__icontains=q))
    items = items.order_by('-created_at')[:200]
    return render(request, 'index.html', {
        'items': items,
        'q': q,
        'is_admin': is_admin(request.user),
    })


@api_login_required
@require_POST
def upload_audio(request):
    audio_file = request.FILES.get('audio')
    if not audio_file:
        return JsonResponse({'error': 'فایلی ارسال نشد'}, status=400)
    ext = Path(audio_file.name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return JsonResponse({'error': 'فرمت فایل پشتیبانی نمی‌شود'}, status=400)

    patient_name, patient_nid, err = _clean_patient(request.POST)
    if err:
        return JsonResponse({'error': err}, status=400)

    lang = request.POST.get('language', DEFAULT_LANG)
    if lang not in LANGS:
        lang = DEFAULT_LANG

    _cleanup_old()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    path = TMP_DIR / f'{job_id}{ext}'
    with open(path, 'wb') as f:
        for chunk in audio_file.chunks():
            f.write(chunk)

    with JOBS_LOCK:
        JOBS[job_id] = {
            'status': 'pending', 'text': '', 'error': '',
            'path': str(path), 'filename': audio_file.name, 'created': time.time(),
            'language': lang,
            'owner_id': request.user.id,
            'patient_name': patient_name,
            'patient_national_id': patient_nid,
            'report_html': '', 'ai_status': '', 'ai_error': '',
        }
    threading.Thread(target=_process, args=(job_id,), daemon=True).start()
    return JsonResponse({'id': job_id, 'status': 'pending'})


@api_login_required
def status(request, job_id):
    data = _get_job(request, job_id)
    if not data:
        return JsonResponse({'status': 'failed', 'error': 'این تبدیل پیدا نشد (ممکن است سرور ری‌استارت شده باشد)'}, status=404)
    return JsonResponse({
        'status': data['status'], 'error': data['error'],
        'ai_status': data.get('ai_status', ''), 'ai_error': data.get('ai_error', ''),
    })


@login_required
def result(request, job_id):
    job = _get_job(request, job_id)
    if not job or job['status'] != 'done':
        return redirect('index')

    has_report = bool(job.get('report_html'))
    is_ai = has_report and request.GET.get('view') != 'original'
    content_html = job['report_html'] if is_ai else linebreaks_filter(job['text'], autoescape=True)

    return render(request, 'result.html', {
        'mode': 'job',
        'job_id': job_id,
        'title': f"{job['patient_name']} - {job['patient_national_id']}",
        'patient_name': job['patient_name'],
        'patient_national_id': job['patient_national_id'],
        'content_html': content_html,
        'is_ai': is_ai,
        'has_report': has_report,
        'can_edit': True,
        'is_owner': True,
        'save_url': reverse('save_result', args=[job_id]),
        'discard_url': reverse('discard_result', args=[job_id]),
        'audio_url': '',
        'lang': job['language'],
        'dir': LANGS[job['language']]['dir'],
    })


@api_login_required
@require_POST
def save_result(request, job_id):
    key = str(job_id)
    job = _get_job(request, key)
    if not job or job['status'] != 'done':
        return JsonResponse({'error': 'این نتیجه دیگر در دسترس نیست'}, status=404)
    try:
        html = json.loads(request.body).get('html', '')
    except (ValueError, AttributeError):
        return JsonResponse({'error': 'درخواست نامعتبر'}, status=400)

    obj = Transcription(
        owner=request.user,
        patient_name=job['patient_name'],
        patient_national_id=job['patient_national_id'],
        text=sanitize_html(html),
        status='done',
        language=job['language'],
    )
    with open(job['path'], 'rb') as f:
        obj.audio_file.save(job['filename'], File(f), save=False)
    obj.save()

    with JOBS_LOCK:
        JOBS.pop(key, None)
    _remove_tmp(job)
    return JsonResponse({'url': reverse('detail', args=[obj.id])})


@api_login_required
@require_POST
def discard_result(request, job_id):
    key = str(job_id)
    if not _get_job(request, key):
        return JsonResponse({'url': reverse('index')})
    with JOBS_LOCK:
        job = JOBS.pop(key, None)
    if job:
        _remove_tmp(job)
    return JsonResponse({'url': reverse('index')})


@login_required
def detail(request, pk):
    obj = get_object_or_404(
        Transcription.objects.visible_to(request.user).select_related('owner'), pk=pk)
    is_owner = obj.owner_id == request.user.id

    ctx = {
        'mode': 'saved',
        'obj_id': obj.id,
        'title': obj.display_title,
        'patient_name': obj.patient_name,
        'patient_national_id': obj.patient_national_id,
        'content_html': obj.text,
        'can_edit': is_owner,      # فقط صاحب ویس؛ مدیر و پزشکان دیگر فقط مشاهده
        'is_owner': is_owner,
        'owner_name': display_name(obj.owner),
        'save_url': reverse('update_saved', args=[pk]) if is_owner else '',
        'discard_url': '',
        'audio_url': reverse('audio', args=[pk]),
        'lang': obj.language,
        'dir': LANGS.get(obj.language, LANGS['fa'])['dir'],
    }
    if is_owner:
        ctx.update({
            'doctors': _doctors_qs(exclude_user=request.user),
            'shared_all': obj.shared_with_all,
            'shared_ids': list(obj.shared_with.values_list('id', flat=True)),
        })
    return render(request, 'result.html', ctx)


@api_login_required
@require_POST
def update_saved(request, pk):
    obj = get_object_or_404(Transcription.objects.visible_to(request.user), pk=pk)
    if obj.owner_id != request.user.id:
        return JsonResponse({'error': 'شما اجازه ویرایش این مورد را ندارید'}, status=403)
    try:
        html = json.loads(request.body).get('html', '')
    except (ValueError, AttributeError):
        return JsonResponse({'error': 'درخواست نامعتبر'}, status=400)
    obj.text = sanitize_html(html)
    obj.save(update_fields=['text', 'updated_at'])
    return JsonResponse({'ok': True})


@api_login_required
@require_POST
def share_saved(request, pk):
    """فقط صاحب ویس می‌تواند اشتراک‌گذاری را تغییر دهد."""
    obj = get_object_or_404(Transcription, pk=pk, owner=request.user)
    try:
        body = json.loads(request.body)
        share_all = bool(body.get('all'))
        ids = [int(i) for i in (body.get('doctor_ids') or [])]
    except (ValueError, TypeError, AttributeError):
        return JsonResponse({'error': 'درخواست نامعتبر'}, status=400)

    doctors = _doctors_qs(exclude_user=request.user).filter(pk__in=ids)
    obj.shared_with_all = share_all
    obj.save(update_fields=['shared_with_all', 'updated_at'])
    obj.shared_with.set(doctors)
    return JsonResponse({'ok': True, 'all': share_all, 'count': doctors.count()})


@api_login_required
@require_POST
def start_report(request, job_id):
    key = str(job_id)
    try:
        body = json.loads(request.body)
    except (ValueError, AttributeError):
        return JsonResponse({'error': 'درخواست نامعتبر'}, status=400)
    text = (body.get('text') or '').strip()[:5000]
    doctor_prompt = (body.get('prompt') or '').strip()[:2000]
    if not text:
        return JsonResponse({'error': 'متنی برای ارسال وجود ندارد'}, status=400)

    with JOBS_LOCK:
        job = JOBS.get(key)
        if not job or job.get('owner_id') != request.user.id or job['status'] != 'done':
            return JsonResponse({'error': 'این نتیجه دیگر در دسترس نیست'}, status=404)
        if job.get('ai_status') == 'generating':
            return JsonResponse({'error': 'گزارش در حال ساخت است'}, status=409)
        job['ai_status'] = 'generating'
        job['ai_error'] = ''

    threading.Thread(target=_generate, args=(key, text, doctor_prompt), daemon=True).start()
    return JsonResponse({'status': 'generating'})

USERNAME_RE = re.compile(r'^[\w.@+-]{3,150}$')


def api_admin_required(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'نشست شما منقضی شده؛ دوباره وارد شوید'}, status=401)
        if not is_admin(request.user):
            return JsonResponse({'error': 'فقط مدیر به این بخش دسترسی دارد'}, status=403)
        return view(request, *args, **kwargs)
    return wrapper


def _doctor_or_404(pk):
    return get_object_or_404(User, pk=pk, is_staff=False, is_superuser=False)


def _check_password(password, user=None):
    try:
        validate_password(password, user=user)
    except ValidationError as e:
        return e.messages[0]
    return None


@login_required
def doctors_admin(request):
    if not is_admin(request.user):
        raise PermissionDenied
    doctors = (User.objects.filter(is_staff=False, is_superuser=False)
               .annotate(n_items=Count('transcriptions'))
               .order_by('-is_active', 'first_name', 'last_name', 'username'))
    return render(request, 'doctors.html', {'doctors': doctors})


@api_admin_required
@require_POST
def doctor_create(request):
    try:
        body = json.loads(request.body)
    except (ValueError, AttributeError):
        return JsonResponse({'error': 'درخواست نامعتبر'}, status=400)

    first = re.sub(r'\s+', ' ', str(body.get('first_name') or '')).strip()
    last = re.sub(r'\s+', ' ', str(body.get('last_name') or '')).strip()
    username = str(body.get('username') or '').strip()
    password = str(body.get('password') or '')

    if not first or not last:
        return JsonResponse({'error': 'نام و نام خانوادگی را وارد کنید'}, status=400)
    if len(first) > 150 or len(last) > 150:
        return JsonResponse({'error': 'نام یا نام خانوادگی خیلی طولانی است'}, status=400)
    if not USERNAME_RE.match(username):
        return JsonResponse(
            {'error': 'نام کاربری باید ۳ تا ۱۵۰ نویسه و فقط شامل حروف انگلیسی، عدد و . @ + - _ باشد'}, status=400)
    if User.objects.filter(username__iexact=username).exists():
        return JsonResponse({'error': 'این نام کاربری قبلاً ثبت شده است'}, status=400)

    probe = User(username=username, first_name=first, last_name=last)
    err = _check_password(password, probe)
    if err:
        return JsonResponse({'error': err}, status=400)

    user = User.objects.create_user(
        username=username, password=password,
        first_name=first, last_name=last,
        is_staff=False, is_superuser=False, is_active=True,
    )
    return JsonResponse({'ok': True, 'id': user.id})


@api_admin_required
@require_POST
def doctor_toggle(request, pk):
    user = _doctor_or_404(pk)
    user.is_active = not user.is_active
    user.save(update_fields=['is_active'])
    return JsonResponse({'ok': True, 'is_active': user.is_active})


@api_admin_required
@require_POST
def doctor_password(request, pk):
    user = _doctor_or_404(pk)
    try:
        password = str(json.loads(request.body).get('password') or '')
    except (ValueError, AttributeError):
        return JsonResponse({'error': 'درخواست نامعتبر'}, status=400)
    err = _check_password(password, user)
    if err:
        return JsonResponse({'error': err}, status=400)
    user.set_password(password)
    user.save(update_fields=['password'])
    return JsonResponse({'ok': True})
