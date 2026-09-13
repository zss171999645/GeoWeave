# 用于自动发送4dlabel计算资源使用情况的消息；

from datetime import datetime, timedelta
import time
import requests
import json
import subprocess
import re
import pandas as pd


CLUSTER_NAMES = [
    "project-a800-4dlabel-perception-bcloud",
    "project-a800-4dlabel-perception2-bcloud",
    "share-a800-small-bcloud",
    "project-l20-4dlabel-perception-acloud",
    "project-l20-4dlabel-perception-tcloud",
]

FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/7f1c8ad7-a225-4fb4-8e43-ae66b4859560"


def run_command(command, show_output=False):
    try:
        # print("call", ' '.join(command) if isinstance(command, list) else command)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE if not show_output else None,
            stderr=subprocess.PIPE if not show_output else None,
            text=True if not show_output else None,       # 以字符串形式返回输出
            check=True,       # 如果命令返回非零退出状态，将引发CalledProcessError
            shell=True if isinstance(command, str) else False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to run command: {e.stderr}")
        return None

def parse_table_with_regex(output):
    """
    使用正则表达式解析表格输出，并返回包含字典的列表。
    
    :param output: 命令输出的表格字符串
    :return: 字典列表，每个字典对应表格中的一行数据
    """
    if not output:
        return []
    lines = output.strip().splitlines()
    data = []
    headers = []
    
    # 正则表达式匹配以 '|' 开头和结尾的行，并捕获中间的字段
    pattern = re.compile(r'^\|(.+)\|$')
    
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue  # 跳过不符合的行（如分隔线）
        
        # 使用 re.split 分割字段，并去除多余的空格
        fields = [field.strip() for field in match.group(1).split('|')]
        
        if not headers:
            headers = fields  # 第一行符合的是表头
            continue
        
        if len(fields) != len(headers):
            print("字段数与标题数不匹配，跳过该行")
            continue
        
        # 创建字典，键为表头，值为对应字段
        entry = dict(zip(headers, fields))
        data.append(entry)
    
    return data


def try_catch_wrapper(func):
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return None
    return wrapped_func


@try_catch_wrapper
def query_gpu_queue():
    entries = []
    for cluster_name in CLUSTER_NAMES:
        cmd = [
            "aidi-inf-cli",
            "job",
            "quota",
            "--queue_name",
            cluster_name,
        ]
        output = run_command(cmd)
        data = parse_table_with_regex(output)
        for entry in data:
            entry["is_free"] = int(entry["QUEUING JOB"]) == 0 and int(entry["FREE QUOTA"]) > 0
        entries.extend(data)
    df = pd.DataFrame(entries)
    # sorted by int(QUEUING JOB)
    df = df.sort_values(by="is_free", ascending=False)
    df = df.sort_values(by="QUEUING JOB", key=lambda x: x.astype(int), ascending=True)
    df_string = ""
    for _, row in df.iterrows():
        df_string += f"{row['QUEUE NAME']} \n  运行/排队任务数: {row['RUNNING JOB']}/{row['QUEUING JOB']} {'空卡: ' if row['is_free'] else '不可用: '} {row['FREE QUOTA']}/{row['ALLOCATE QUOTA']} \n"

    # if any is_free is True, return str(df) else return None
    has_free_gpu = df["is_free"].any()
    any_no_queue_cluster = (df["QUEUING JOB"] == "0").any()
    if has_free_gpu or any_no_queue_cluster:
        if has_free_gpu:
            cluster_with_free_gpu = df[df["is_free"]]["QUEUE NAME"].unique()
            msg = f"\n WARNING: 有空余资源：\n"
            for cluster in cluster_with_free_gpu:
                n_free_gpu = df[df["QUEUE NAME"] == cluster]["FREE QUOTA"].values[0]
                msg += f"{cluster} 空余{n_free_gpu}张GPU\n"
            msg += "\n"
        else:
            return None
        return "4dlabel计算资源" + "\n" + df_string + "\n" + msg
    else:
        return None


def send_message(message):
    data = {
        "msg_type": "text",
        "content": {
            "text": message
        }
    }
    response = requests.post(FEISHU_WEBHOOK_URL, data=json.dumps(data), headers={"Content-Type": "application/json"})
    return response


if __name__ == "__main__":
    # query every 15 minites during daytime 8am~11pm
    # if |current_time - last_send_time| > 1 hour, send message
    # if |current_time - last_send_time| < 1 hour, sleep 15 minites

    last_send_time = datetime.now() - timedelta(hours=1)
    while True:
        now = datetime.now()
        if (now - last_send_time).total_seconds() >= 3600 and now.hour >= 8 and now.hour <= 23:
            message = query_gpu_queue()
            if message:
                print(f"going to send message: \n----\n{message}\n----\n")
                response = send_message(message)
                print("sent message", response.status_code, response.text)
                last_send_time = now
            else:
                print("no free gpu, sleep 15 minites")
        else:
            print("waiting 15 minites")
        time.sleep(900)