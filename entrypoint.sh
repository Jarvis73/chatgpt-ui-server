#!/bin/bash

# Create new migration files (when create new Model)
python manage.py makemigrations

# Apply new migration files to the database
python manage.py migrate

python manage.py createsuperuser --no-input

export WORKERS=${SERVER_WORKERS:-3}

export TIMEOUT=${WORKER_TIMEOUT:-180}

exec gunicorn chatgpt_ui_server.wsgi --workers=$WORKERS --timeout $TIMEOUT --bind 0.0.0.0:8000 --access-logfile - --worker-class gevent