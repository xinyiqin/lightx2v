# latest RabbitMQ 4.x
docker run -it --rm \
    --name rabbitmq \
    --hostname test-rabbit-node \
    -e RABBITMQ_NODENAME=rabbit@test-rabbit-node \
    -e RABBITMQ_DEFAULT_USER=lightx2v \
    -e RABBITMQ_DEFAULT_PASS=lightx2v \
    -p 5672:5672 \
    -p 15672:15672 \
    -v /data/nvme1/liuliang1/lightx2v/local_rabbitmq:/var/lib/rabbitmq \
    rabbitmq:4-management
