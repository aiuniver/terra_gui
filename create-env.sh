#!/bin/bash

cat <<EOT >> .env
SECRET_KEY=6plk8y@aj%7^1ve#d(m+x+)5y15^&d(7$=0s+ju&i-kj+19q
DEBUG=True
ALLOWED_HOSTS=.terra.neural-university.ru
TERRA_AI_EXCHANGE_API_URL=http://bl146u.xyz:8099/api/v1/exchange
EOT
