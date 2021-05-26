rm file.zip
find . -path '*/.*' -prune -o -type f -print | zip file.zip -@
