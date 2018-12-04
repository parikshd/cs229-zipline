import boto3
import botocore

# Let's use Amazon S3
s3 = boto3.resource('s3')

BUCKET_NAME = 'nest-1-ap-south-1'
prefix = 'flight_logs/03/12/2018/'

client = boto3.client('s3')
result = client.list_objects(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter='/')
for o in result.get('CommonPrefixes'):
    orig_key = o.get('Prefix')
    flight_id = orig_key.split("nest_1_flight_", 1)[1]
    flight_id = flight_id.rstrip('\/')
    final_key = orig_key + "analysis/flight.yaml"
    try:
        s3.Bucket(BUCKET_NAME).download_file(final_key, 'testinput/03122018/' + flight_id + ".yaml")
        print("Downloaded flight:" + final_key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise