# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2018-06-24 13:52
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.manager


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0006_report'),
    ]

    operations = [
        migrations.CreateModel(
            name='Candle',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(help_text='Qual o seu e-mail?', max_length=200)),
            ],
            managers=[
                ('pdobjects', django.db.models.manager.Manager()),
            ],
        ),
    ]