docker run -it --rm \
    --name postgres \
    -p 5432:5432 \
    -e POSTGRES_USER=lightx2v \
    -e POSTGRES_PASSWORD=lightx2v \
    -e POSTGRES_DB=lightx2v \
    -v /data/nvme1/liuliang1/lightx2v/local_postgres:/var/lib/postgresql/data  \
    postgres:17.5-alpine
