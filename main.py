import boto
import boto.s3.connection
from boto.s3.key import Key

conn = boto.connect_s3(
    aws_access_key_id='LVNWBPUZOVCST4WETUZM',
    aws_secret_access_key='43u4QNnA1NICNZKhlZnOHjJvXQoPwneJ0x4XccOz',
    host='obs.cn-north-4.myhuaweicloud.com', is_secure=False,
    calling_format=boto.s3.connection.OrdinaryCallingFormat())
conn.auth_region_name = 'cn-north-4'
for bucket in conn.get_all_buckets():
        print ("{name}\t{created}".format(
                name = bucket.name,
                created = bucket.creation_date,
        ))
bucket = conn.get_bucket('cifar-tk')
# for bucket in conn.get_all_buckets():
#         print ("{name}\t{created}".format(
#                 name = bucket.name,
#                 created = bucket.creation_date,
#         ))
k = Key(bucket)
filename2 = 'data_batch_5'
k.get_contents_to_filename(filename2)

