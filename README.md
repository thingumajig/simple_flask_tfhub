# simple_flask_tfhub
universal sentence encoder as microservice

curl -i -H "Content-Type: application/json" -X POST -d '@test_data.json' http://localhost:5000/use/api/v1.0/text/elmo
curl -i -H "Content-Type: application/json" -X POST -d '@test_data.json' http://localhost:5000/use/api/v1.0/text/use