import os
import uuid

from django.conf import settings
from django.db import models
from django.db.models import Q


def audio_upload_path(instance, filename):
    ext = os.path.splitext(filename)[1].lower()
    return f'audio_uploads/{uuid.uuid4().hex}{ext}'


class TranscriptionQuerySet(models.QuerySet):
    def visible_to(self, user):
        if user.is_staff or user.is_superuser:
            return self
        return self.filter(
            Q(owner=user) | Q(shared_with_all=True) | Q(shared_with=user)
        ).distinct()


class Transcription(models.Model):
    STATUS_CHOICES = [
        ('pending', 'در انتظار'),
        ('processing', 'در حال پردازش'),
        ('done', 'انجام شده'),
        ('failed', 'خطا'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.PROTECT, related_name='transcriptions',
    )
    patient_name = models.CharField(max_length=150, blank=True)
    patient_national_id = models.CharField(max_length=10, blank=True, db_index=True)

    shared_with_all = models.BooleanField(default=False)
    shared_with = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name='shared_transcriptions',
    )

    audio_file = models.FileField(upload_to=audio_upload_path)
    text = models.TextField(blank=True)
    language = models.CharField(max_length=5)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = TranscriptionQuerySet.as_manager()

    class Meta:
        ordering = ['-created_at']

    @property
    def display_title(self):
        if self.patient_name or self.patient_national_id:
            return f'{self.patient_name} - {self.patient_national_id}'.strip(' -')
        return os.path.basename(self.audio_file.name)

    def __str__(self):
        return f'{self.display_title} ({self.status})'