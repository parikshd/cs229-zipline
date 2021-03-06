import boto3
import botocore
from datetime import datetime
import os

# Let's use Amazon S3
s3 = boto3.resource('s3')

BUCKET_NAME = 'nest-1-ap-south-1'

# currentDay = datetime.now().strftime('%d')
# currentMonth = datetime.now().strftime('%m')
# currentYear = datetime.now().strftime('%Y')

currentDay = '08'
currentMonth = '12'
currentYear = '2018'

outputfolder = str(currentDay)+str(currentMonth)+str(currentYear)
prefix = 'flight_logs/' + str(currentDay) + '/' + str(currentMonth) + '/' + str(currentYear) + '/'

import os
if not os.path.exists('testinput/' + outputfolder):
    os.mkdir('testinput/' + outputfolder)

client = boto3.client('s3')
result = client.list_objects(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter='/')
for o in result.get('CommonPrefixes'):
    orig_key = o.get('Prefix')
    flight_id = orig_key.split("nest_1_flight_", 1)[1]
    flight_id = flight_id.rstrip('\/')
    final_key = orig_key + "analysis/flight.yaml"
    try:
        s3.Bucket(BUCKET_NAME).download_file(final_key, 'testinput/' + outputfolder + '/' + flight_id + ".yaml")
        print("Downloaded flight:" + final_key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise