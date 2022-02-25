cat it_isdt_train_tagged.txt | tr ' ' '\n' | grep -v '^[A-Z]*$' | sort | uniq -c
