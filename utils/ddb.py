import boto3

class ddb_handler():
    
    def __init__(self, strTableName, region_name="ap-northeast-2"):
        
        client = boto3.resource('dynamodb', region_name=region_name)
        self.table = client.Table(strTableName)
        print(f"Table [{strTableName}] is {self.table.table_status} now.")
            
    def put_item(self, dicItem):
        
        self.table.put_item(Item=dicItem)
        
    def get_item(self, dicKey):
        
        response = self.table.get_item(Key=dicKey)
        if 'Item' in response: return response['Item']
        else: return 'NONE'
        
    def truncate_table(self, ):
        
        #get the table keys
        tableKeyNames = [key.get("AttributeName") for key in self.table.key_schema]
    
        #Only retrieve the keys for each item in the table (minimize data transfer)
        projectionExpression = ", ".join('#' + key for key in tableKeyNames)
        expressionAttrNames = {'#'+key: key for key in tableKeyNames}
        
        counter = 0
        page = self.table.scan(ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames)
        with self.table.batch_writer() as batch:
            while page["Count"] > 0:
                counter += page["Count"]
                # Delete items in batches
                for itemKeys in page["Items"]:
                    batch.delete_item(Key=itemKeys)
                # Fetch the next page
                if 'LastEvaluatedKey' in page:
                    page = table.scan(
                        ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames,
                        ExclusiveStartKey=page['LastEvaluatedKey'])
                else:
                    break
        print(f"Deleted {counter}")
        
class ddb_constructor():
    
    def __init__(self, region_name="ap-northeast-2"):
        
        self.client = boto3.resource('dynamodb', region_name=region_name)
        
    def create_table(self, **kwargs):
        #print (**kwargs)
        
        response = self.client.create_table(**kwargs)
        print (f"{response} was created!!")