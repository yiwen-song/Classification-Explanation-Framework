#!/bin/bash
export FLASK_APP=app.py
export FLASK_ENV=development
rm -rf ./tmp/*
flask run
