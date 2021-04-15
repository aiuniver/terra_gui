run:
	echo "Makefile TerraGUI"

runserver:
	pip install -r ./requirements/colab.txt
	bash ./create-env.sh
	chmod 400 ./tunnel-rsa.key
	python ./manage.py runserver 80 & ssh -i './tunnel-rsa.key' -o StrictHostKeyChecking=no -R $(PORT):localhost:80 test007@labstory.neural-university.ru
