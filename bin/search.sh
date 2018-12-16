cd "$(dirname "$0")"

cd ..


nohup python -u code_felix/core/search.py $1 >> search$1.log 2>&1 &
