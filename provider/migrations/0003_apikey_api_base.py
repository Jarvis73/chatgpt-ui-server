# Generated by Django 4.1.7 on 2023-07-27 09:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('provider', '0002_invcode'),
    ]

    operations = [
        migrations.AddField(
            model_name='apikey',
            name='api_base',
            field=models.CharField(default='https://api.openai.com/v1', max_length=100),
        ),
    ]
