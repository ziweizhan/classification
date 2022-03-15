
## 1. web部署

web部署就是采用REST API的形式进行接口调用。

web部署的方式采用flask+ redis的方法进行模型部署,pytorch为模型的框架,flask为后端框架,redis是采用键值的形式存储图像的数据库。



各package包的版本：

```
pytorch                   1.2.0
flask                     1.0.2 
Redis			  3.0.6
```



### 1. Redis安装,配置

ubuntu Redis的安装,下载地址:https://redis.io/download

安装教程: https://www.jianshu.com/p/bc84b2b71c1c

```shell
wget http://download.redis.io/releases/redis-6.0.6.tar.gz
# 拷贝到/usr/local目录下
cp redis-3.0.0.rar.gz /usr/local
# 解压
tar xzf redis-6.0.6.tar.gz

cd /usr/local/redis-6.0.6

# 安装至指定的目录下
make PREFIX=/usr/local/redis install
```



Redis配置:

```shell
# redis.conf是redis的配置文件，redis.conf在redis源码目录。
# 拷贝配置文件到安装目录下
# 进入源码目录，里面有一份配置文件 redis.conf，然后将其拷贝到安装路径下
cd /usr/local/redis
cp /usr/local/redis-3.0.0/redis.conf  /usr/local/redis/bin
```

此时在/usr/local/redis/bin目录下,有如下文件:

```shell
redis-benchmark redis性能测试工具
redis-check-aof AOF文件修复工具
redis-check-rdb RDB文件修复工具
redis-cli redis命令行客户端
redis.conf redis配置文件
redis-sentinal redis集群管理工具
redis-server redis服务进程
```

Redis服务开启:

```shell
# 这是以前端方式启动,关闭终端,服务停止
./redis-server

# 后台方式启动
#修改redis.conf配置文件， daemonize yes 以后端模式启动

cd /usr/local/redis
./bin/redis-server ./redis.conf
```



连接Redis

```shell
/usr/local/redis/bin/redis-cli 
```

关闭Redis

```shell
cd /usr/local/redis
./bin/redis-cli shutdown
```

强行中止Redis,(可能会丢失持久化数据)

```shell
pkill redis-server
```



### 2. server端



```python
@app.route('/predict', methods=['POST'])
def predict():

    data = {'Success': False}

    if request.files.get('image'):

        now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

        image = request.files['image'].read()
        image = Image.open(io.BytesIO(image))
        image = image_transform(InputSize)(image).numpy()
        # 将数组以C语言存储顺序存储
        image = image.copy(order="C")
        # 生成图像ID
        k = str(uuid.uuid4())
        d = {"id": k, "image": base64_encode_image(image)}
        # print(d)
        db.rpush(ImageQueue, json.dumps(d))
        # 运行服务
        while True:
            # 获取输出结果
            output = db.get(k)
            # print(output)
            if output is not None:
                output = output.decode("utf-8")
                data["predictions"] = json.loads(output)
                db.delete(k)
                break
            time.sleep(ClientSleep)
        data["success"] = True
    return jsonify(data)

if __name__ == '__main__':

    app.run(host='127.0.0.1', port =5000,debug=True )
```



### 3. Redis服务器端

```python
def classify_process(filepath):
    # 导入模型
    print("* Loading model...")
    model = load_checkpoint(filepath)
    print("* Model loaded")
    while True:
        # 从数据库中创建预测图像队列
        queue = db.lrange(ImageQueue, 0, BatchSize - 1)
        imageIDs = []
        batch = None
        # 遍历队列
        for q in queue:
            # 获取队列中的图像并反序列化解码
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], ImageType,
                                        (1, InputSize[0], InputSize[1], Channel))
            # 检查batch列表是否为空
            if batch is None:
                batch = image
            # 合并batch
            else:
                batch = np.vstack([batch, image])
            # 更新图像ID
            imageIDs.append(q["id"])
            # print(imageIDs)
        if len(imageIDs) > 0:
            print("* Batch size: {}".format(batch.shape))
            preds = model(torch.from_numpy(batch.transpose([0, 3,1,2])))
            results = decode_predictions(preds)
            # 遍历图像ID和预测结果并打印
            for (imageID, resultSet) in zip(imageIDs, results):
                # initialize the list of output predictions
                output = []
                # loop over the results and add them to the list of
                # output predictions
                print(resultSet)
                for label in resultSet:
                    prob = label.item()
                    r = {"label": label.item(), "probability": float(prob)}
                    output.append(r)
                # 保存结果到数据库
                db.set(imageID, json.dumps(output))
            # 从队列中删除已预测过的图像
            db.ltrim(ImageQueue, len(imageIDs), -1)
        time.sleep(ServeSleep)




def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


if __name__ == '__main__':
    filepath = '../c/resnext101_32x8.pth'
    classify_process(filepath)
```



### 4. 调用测试



```shell
curl -X POST -F image=@test.jpg 'http://127.0.0.1:5000/predict'
```



```python
from threading import Thread
import requests
import time

# 请求的URL
REST_API_URL = "http://127.0.0.1:5000/predict"
# 测试图片
IMAGE_PATH = "./test.jpg"

# 并发数
NUM_REQUESTS = 500
# 请求间隔
SLEEP_COUNT = 0.05
def call_predict_endpoint(n):

    # 上传图像
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}
    # 提交请求
    r = requests.post(REST_API_URL, files=payload).json()
    # 确认请求是否成功
    if r["success"]:
        print("[INFO] thread {} OK".format(n))
    else:
        print("[INFO] thread {} FAILED".format(n))
# 多线程进行
for i in range(0, NUM_REQUESTS):
    # 创建线程来调用api
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)
time.sleep(300)
```