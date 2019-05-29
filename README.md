# simple_flask_tfhub
universal sentence encoder as microservice

## usage
ELMO(size 1024):
```
curl -i -H "Content-Type: application/json" -X POST -d '@test_data.json' http://localhost:8080/use/api/v1.0/text/elmo
```
Universal Sentence Encoding(size 512):
```
curl -i -H "Content-Type: application/json" -X POST -d '@test_data.json' http://localhost:8080/use/api/v1.0/text/use
```
```
curl -i -H "Content-Type: application/json" -X POST -d '@test_data.json' http://xxxxx.compute.amazonaws.com:8080/use/api/v1.0/text/use
```

Send file:
```
curl -i  -H "Content-Type: multipart/form-data" -X POST -F text=@test_file.txt http://localhost:8080/use/api/v1.0/text/use
```

## service commands
### service configuration
```
sudo cp tfhub_microservice.service /lib/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tfhub_microservice.service
sudo systemctl start tfhub_microservice.service
```
### service status
`sudo systemctl status tfhub_microservice.service`

### service commands
```
sudo systemctl stop tfhub_microservice.service          #To stop running service 
sudo systemctl start tfhub_microservice.service         #To start running service 
sudo systemctl restart tfhub_microservice.service       #To restart running service 
```
