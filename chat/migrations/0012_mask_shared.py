# Generated by Django 4.2.2 on 2023-06-18 04:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0011_conversation_mask_conversation_mask_avatar_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='mask',
            name='shared',
            field=models.BooleanField(default=False),
        ),
    ]