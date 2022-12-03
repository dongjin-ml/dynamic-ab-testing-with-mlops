import os
import json
import boto3
import random
from ddb import ddb_handler
from decimal import Decimal
from operator import itemgetter
from datetime import datetime, timezone, timedelta

DDB_ASSIGNMENT=os.environ['DDB_ASSIGNMENT']
DDB_METRIC=os.environ['DDB_METRIC']
DDB_METRIC_VISUALIZATION=os.environ['DDB_METRIC_VISUALIZATION']
DDB_HELPFUL_REVIEWS=os.environ['DDB_HELPFUL_REVIEWS']
REGION=os.environ['REGION']
MODEL_ENDPOINT=os.environ['MODEL_ENDPOINT']
WARM_CNT=int(os.environ['WARM_CNT'])

# class ddb_handler():
    
#     def __init__(self, strTableName):
        
#         client = boto3.resource('dynamodb', region_name="ap-northeast-2")
#         self.table = client.Table(strTableName)
#         print(f"Table [{strTableName}] is {self.table.table_status} now.")
        
#     def put_item(self, dicItem):
        
#         self.table.put_item(Item=dicItem)
        
#     def get_item(self, dicKey):
        
#         response = self.table.get_item(Key=dicKey)
#         if 'Item' in response: return response['Item']
#         else: return 'NONE'
        
#     def truncate_table(self, ):
        
#         #get the table keys
#         tableKeyNames = [key.get("AttributeName") for key in self.table.key_schema]
    
#         #Only retrieve the keys for each item in the table (minimize data transfer)
#         projectionExpression = ", ".join('#' + key for key in tableKeyNames)
#         expressionAttrNames = {'#'+key: key for key in tableKeyNames}
        
#         counter = 0
#         page = self.table.scan(ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames)
#         with self.table.batch_writer() as batch:
#             while page["Count"] > 0:
#                 counter += page["Count"]
#                 # Delete items in batches
#                 for itemKeys in page["Items"]:
#                     batch.delete_item(Key=itemKeys)
#                 # Fetch the next page
#                 if 'LastEvaluatedKey' in page:
#                     page = table.scan(
#                         ProjectionExpression=projectionExpression, ExpressionAttributeNames=expressionAttrNames,
#                         ExclusiveStartKey=page['LastEvaluatedKey'])
#                 else:
#                     break
#         print(f"Deleted {counter}")
            
class prediction():
    
    def __init__(self, ):
        
        self.runtime = boto3.Session().client('sagemaker-runtime')
    
    def _chunker(self, seq, nBatchSize):
        return (seq[pos:pos + nBatchSize] for pos in range(0, len(seq), nBatchSize))
    
    def _parse_predictions(self, results):
        # return [(result['label'][0] == '__label__Helpful', result['prob'][0]) for result in results]
        return [(result['label'][0] == '__label__Helpful', result['prob'][0]) \
                if result['label'][0] == '__label__Helpful' \
                else (result['label'][0] == '__label__Helpful', 1 - result['prob'][0]) for result in results]   
    
    def predict(self, endpoint_name, variant_name, data, nBatchSize=50):
        
        '''
        boto3 API, "invoke_endpoint", 를 호출하여 예측 결과를 얻은 후에 레이블 TRUE/FALSE 및 confidense score 를 제공 
        '''
        all_predictions = []
        for array in self._chunker(data, nBatchSize):
            payload = {"instances" : array, "configuration": {"k": 1} }
            try:
                response = self.runtime.invoke_endpoint(
                    EndpointName = endpoint_name, 
                    TargetVariant = variant_name,
                    ContentType = 'application/json',                        
                    Body = json.dumps(payload))
                predictions = json.loads(response['Body'].read())            
                all_predictions += self._parse_predictions(predictions)
            except Exception as e:
                print(e)
                print(payload)
                
        return all_predictions
    
    def join_results(self, test_df, predictions, variant_name):
        
        pred_df = pd.DataFrame(predictions, columns=['is_helpful_prediction', 'is_helpful_prob'])
        pred_df['variant_name'] = variant_name
        
        return test_df[['is_helpful']].join(pred_df)

class abtest_api():
    
    def __init__(self, ):
        
        self.strEndPointName = MODEL_ENDPOINT#"helpfulness-detector-endpoint-2022-11-29-09-51-34"
        self.predictor = prediction()
        self.nRandomEvent = WARM_CNT#150
        self.strSelectedVariantName = 'variant-A'
        
        self.ddb_assignment = ddb_handler(strTableName=DDB_ASSIGNMENT)
        self.ddb_helpful_review = ddb_handler(strTableName=DDB_HELPFUL_REVIEWS)
        self.ddb_metric = ddb_handler(strTableName=DDB_METRIC)
        self.ddb_metric_visual = ddb_handler(strTableName=DDB_METRIC_VISUALIZATION)
                
        self.ddb_assignment.truncate_table()
        self.ddb_metric.truncate_table()
        self.ddb_metric_visual.truncate_table()
        
    def _helpfulness_detection(self, product_id, variant_name, review_id, score):
    
        dicKey = {
            'product_id': product_id,
            'variant_name': variant_name
        }
        
        if score >= 0.5:
            
            dicRes = self.ddb_helpful_review.get_item(dicKey)
            
            if dicRes != 'NONE': listHelpfulReviewIDs = dicRes['helpful_review_ids'] 
            else: listHelpfulReviewIDs = []
            
            #listHelpfulReviewIDs.append(review_id)
            #listHelpfulReviewIDs = list(set(listHelpfulReviewIDs))
            
            listHelpfulReviewIDs = list(map(lambda x:tuple(x), listHelpfulReviewIDs))
            listHelpfulReviewIDs.append((review_id, Decimal(str(score))))
            listHelpfulReviewIDs = list(set(listHelpfulReviewIDs))
            listHelpfulReviewIDs = sorted(listHelpfulReviewIDs, key=itemgetter(1), reverse=True)[:5]
            
            dicItem = {
                'product_id': product_id,
                'variant_name': variant_name,
                'helpful_review_ids': listHelpfulReviewIDs
            }
            self.ddb_helpful_review.put_item(dicItem)
        
        else: listHelpfulReviewIDs = None
            
        return listHelpfulReviewIDs
    
    def inference(self, event):
        
        bPred_A, fScore_A = self.predictor.predict(self.strEndPointName, 'variant-A', [event['review']], nBatchSize=1)[0]
        bPred_B, fScore_B = self.predictor.predict(self.strEndPointName, 'variant-B', [event['review']], nBatchSize=1)[0]
        
        #print (f"review_id: {event['review_id']} \nproduct_id: {event['product_id']} \nreview: {event['review']}")
        #print (f"score_A: {fScore_A} \nscore_B: {fScore_B}")
        #print ("===")
        
        listHelpfulReviewIDs_A = self._helpfulness_detection(event['product_id'], 'variant-A', event['review_id'], fScore_A)
        listHelpfulReviewIDs_B = self._helpfulness_detection(event['product_id'], 'variant-B', event['review_id'], fScore_B)
        
        res = {
            "review_id": event['review_id'],
            "product_id": event['product_id'],
            "score_A": fScore_A,
            "score_B": fScore_B,
            "listA": listHelpfulReviewIDs_A,
            "listB": listHelpfulReviewIDs_B,
        }
        
    def invocation(self, event):
        
        # 1. 할당 되어 있는지 확인
        dicKey = {'user_id': event['user_id']}
        dicRes = self.ddb_assignment.get_item(dicKey)
        
        if dicRes != 'NONE': strSelectedVariantName = dicRes['variant_name'] 
        else: strSelectedVariantName = 'NONE'
        
        # 2. 안되었으면 메트릭 가져오기
        if strSelectedVariantName == 'NONE':
            dicMetric = {}
            for strVariantName in ['variant-A', 'variant-B']:
                dicRes = self.ddb_metric.get_item({'variant_name': strVariantName})
                if dicRes != 'NONE': dicMetric[strVariantName] = {'invocation': dicRes['invocation'], 'conversion': dicRes['conversion']}
                else: dicMetric[strVariantName] = {'invocation': 0.0, 'conversion': 0.0}
                
            # 3. 메트릭 기반으로 할당하기 
            listRandomChoice = []
            for strVariantName in ['variant-A', 'variant-B']:
                if dicMetric[strVariantName]['invocation'] < self.nRandomEvent: listRandomChoice.append(True) # 인보케이션이 둘중 하나라도 100회 이상 안되면 
                else: listRandomChoice.append(False)
            
            if all(listRandomChoice):
                strSelectedVariantName = random.choice(['variant-A', 'variant-B'])
            else:
                for strVariantName in ['variant-A', 'variant-B']:
                    dicMetric[strVariantName]['success'] = int(dicMetric[strVariantName]['conversion'])
                    dicMetric[strVariantName]['failure'] = int(dicMetric[strVariantName]['invocation'] - dicMetric[strVariantName]['success'])
            
                listProbSampled = []
                for strVariantName in ['variant-A', 'variant-B']:
                    nSuccess = dicMetric[strVariantName]['success']
                    nFailure = dicMetric[strVariantName]['failure']
                    listProbSampled.append(random.betavariate(nSuccess+1, nFailure+1))
    
                strSelectedVariantName = ['variant-A', 'variant-B'][max(range(len(listProbSampled)), key=lambda x: listProbSampled[x])]
                #print ("method", 'MAB', listProbSampled)
            #print ("dicMetric", dicMetric)
            #print ("strVariantName", strSelectedVariantName)
                
            # 4. assignment 업데이트 하기 
            dicItem = {
                'user_id': event['user_id'],
                'variant_name': strSelectedVariantName 
            }
            self.ddb_assignment.put_item(dicItem)
            
            # 5. 메트릭 invocation 업데이트 하기
            dicItem = {
                'variant_name': strSelectedVariantName,
                'invocation': int(dicMetric[strSelectedVariantName]['invocation']+1),
                'conversion': int(dicMetric[strSelectedVariantName]['conversion'])
            }
            self.ddb_metric.put_item(dicItem)
            
            bTimeTrigger, strTime = time_mngt.time_trigger()
            
            if bTimeTrigger:
                
                nTotalEvent = 0
                for strVariantName in ['variant-A', 'variant-B']:
                    nTotalEvent += int(dicMetric[strVariantName]['invocation'])
                    
                for strVariantName in ['variant-A', 'variant-B']:
                    dicMetric[strVariantName]['success'] = int(dicMetric[strVariantName]['conversion'])
                    dicMetric[strVariantName]['failure'] = int(dicMetric[strVariantName]['invocation'] - dicMetric[strVariantName]['success'])
                    
                    if nTotalEvent != 0: fTrafficDist = int(dicMetric[strVariantName]['invocation'])/nTotalEvent
                    else: fTrafficDist = 0.0
                        
                    fConversionRate = int(dicMetric[strVariantName]['conversion'])/int(dicMetric[strVariantName]['invocation'])
                    
                    dicItem = {
                        'time_stamp':strTime,
                        'variant_name': strVariantName,
                        'invocation': int(dicMetric[strVariantName]['invocation']),
                        'conversion': int(dicMetric[strVariantName]['conversion']),
                        'traffic_distribution': fTrafficDist,
                        'conversion_rate': fConversionRate
                    }
                    dicItem = json.loads(json.dumps(dicItem), parse_float=Decimal)
                    self.ddb_metric_visual.put_item(dicItem)
                
            #return bTimeTrigger, strTime
            #self.strSelectedVariantName = strSelectedVariantName
            
    def conversion(self, event):
        
        bConversion = False
        strSelectedVariantName = self.ddb_assignment.get_item({'user_id': event['user_id']})['variant_name']
        
        try:
            listHelpfulReviews =self.ddb_helpful_review.get_item({'product_id': event['product_id'], 'variant_name': strSelectedVariantName})['helpful_review_ids']
            listHelpfulReviews = [element[0] for element in listHelpfulReviews]
        except: listHelpfulReviews = []
        
        if event['review_id'] in listHelpfulReviews: bConversion = True
        if bConversion:
            dicRes = self.ddb_metric.get_item({'variant_name': strSelectedVariantName})
            dicRes['conversion'] = int(dicRes['conversion']+1)
            self.ddb_metric.put_item(dicRes)

class time_management():
    
    def __init__(self, nInterval=1):
        
        self.nPreviosTriggerMin = 999
        self.nInterval = nInterval
        self.timezone_kst = timezone(timedelta(hours=9))
        
    def time_trigger(self, ):
        
        bTrigger = False
        nMin = datetime.now()
        
        datetime_utc = datetime.utcnow()
        datetime_kst = datetime_utc.astimezone(self.timezone_kst)
        nMin = datetime_kst.minute
        
        if self.nPreviosTriggerMin != nMin and nMin % self.nInterval == 0:
            bTrigger, self.nPreviosTriggerMin = True, nMin
            
        return bTrigger, datetime_kst.strftime('%Y-%m-%dT%H:%M')



time_mngt = time_management(nInterval=1)
api = abtest_api()

def lambda_handler(event, context):

    #if not 'generator_idx' in event or not 'review_id' in event or not 'product_id' in event or not 'review' in event:
    #    return {
    #        'statusCode': 200,
    #        'body': json.dumps('Error: Please specify all parameters (length, n_capitals, n_numbers).')
    #    }
    
    bTimeTrigger, strTime = False, 'None'
    
    print (event)
    
    if event["mode"] == 'inference': api.inference(event)
    elif event["mode"] == 'invocation': api.invocation(event)
    elif event["mode"] == 'conversion':api.conversion(event)  
    
    return {
        'statusCode': 200,
        'body': 'good'
    }