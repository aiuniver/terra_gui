run:
	echo "Makefile TerraGUI"

env:
	chmod +x ./create-env.sh
	/bin/bash ./create-env.sh

runserver:
	chmod 400 ./tunnel-rsa.key
	chmod +x ./manage.py
	python ./manage.py runserver 80 & ssh -i './tunnel-rsa.key' -o StrictHostKeyChecking=no -R $(PORT):localhost:80 test007@labstory.neural-university.ru
