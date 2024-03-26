#!/bin/bash
git config --global user.name "Wu Junde"
git config --global user.email "izzy843794947@gmail.com"
# Add changes to the staging area
git add .

# Commit changes with a default message
git commit -m "update"

# Push changes to the remote repository
git push origin master
