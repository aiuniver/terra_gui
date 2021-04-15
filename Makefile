run:
	echo "Makefile TerraGUI"

env:
	/bin/bash ./create-env.sh

runserver:
	python ./manage.py runserver 80 & ssh -i './tunnel_rsa' -o StrictHostKeyChecking=no -R $(PORT):localhost:80 test007@labstory.neural-university.ru
