import os
import json
import boto3
import argparse
import sagemaker
import numpy as np
import pandas as pd
from datetime import datetime
from spacy.lang.en import English
from botocore.config import Config
from sagemaker.session import production_variant
from sagemaker import session, get_execution_role
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

class evalauator():
    
    def __init__(self, args):
                
        self.args = args
        self.role = self.args.sagemaker_role #self._get_sagemaker_role()
        self.strInOutPrefix = '/opt/ml/processing'
        self.strRegionName = self.args.region # boto3.Session().region_name
        self.sm_client = boto3.client('sagemaker')
        self.sm_session = session.Session(boto3.Session())
        
    def _endpoint(self, strModelPath):
        
        print ("  - Setting endpoint")
        model_image_uri = sagemaker.image_uris.retrieve("blazingtext", self.strRegionName)
        instance_type = "ml.m5.xlarge"
        
        self.sm_session.create_model(name=self.args.model_name,
                                     role=self.role, 
                                     container_defs={"Image": model_image_uri, 
                                                     "ModelDataUrl": strModelPath})
        
        variant_1 = production_variant(model_name=self.args.model_name,
                                       instance_type= instance_type,
                                       initial_instance_count=1,
                                       variant_name="variant-1",
                                       initial_weight=1,)
        
        endpoint_name = f"MAP-Inference-Endpoint-{datetime.now():%Y-%m-%d-%H-%M-%S}"
        self.sm_session.endpoint_from_production_variants(name=endpoint_name, production_variants=[variant_1])
        print(f"EndpointName={endpoint_name} completed!!")
        
        return endpoint_name
    
    def _preprocessing(self, strTestPath):
        
        eval_prep = eval_preprocess(strTestPath)
        pdInput = eval_prep.execution()
        
        return pdInput
    
    def _release_resources(self, strEndPointName):
        
        clean = clean_up()
        clean.delete_endpoint(self.sm_client, strEndPointName ,is_del_model=True)
        
    def _inference(self, listInferenceInput, strEndPointName, strTestPath):
        
        predictor = prediction()
        
        listVariantNames = [pv['VariantName'] for pv \
                            in self.sm_client.describe_endpoint(EndpointName=strEndPointName)['ProductionVariants']]
        listPreds = [predictor.predict(strEndPointName, variant_name, listInferenceInput, nBatchSize=50) for variant_name in listVariantNames]
        
        print (f"listVariantNames: {listVariantNames}")
        print("Size of input: ", len(listInferenceInput))
        print("Prediction Shape: ", np.asarray(listPreds).shape)
        
        pdData = pd.read_csv(strTestPath)
        pdPredResults = pd.concat([predictor.join_results(pdData, predictions, listVariantNames[i]) for (i, predictions) in enumerate(listPreds)])
            
        return pdPredResults
    
    def _get_performance(self, pdPredResults, strOutputPath):
        
        ## Performance 
        pdAccuracy = pdPredResults.groupby('variant_name').apply(lambda g: accuracy_score(g['is_helpful_prediction'], g['is_helpful']))
        fPrec = pdPredResults.query("is_helpful == True and is_helpful_prediction == True").shape[0]/ pdPredResults.query("is_helpful_prediction == True").shape[0] 
        fRec = pdPredResults.query("is_helpful == True and is_helpful_prediction == True").shape[0]/ pdPredResults.query("is_helpful == True").shape[0] 
        fScore = 2 * (fPrec * fRec) / (fPrec + fRec)
        
        ## ROAUC
        for i, (variant_name, pred_df) in enumerate(pdPredResults.groupby('variant_name')):        
            # Get true probability for ROC
            #tp = pred_df.apply(lambda r: r['is_helpful_prob'] if r['is_helpful_prediction'] else 1-r['is_helpful_prob'], axis=1)
            tp = pred_df['is_helpful_prob']
            fpr, tpr, _ = roc_curve(pred_df['is_helpful'], tp)
            fAuc = roc_auc_score(pred_df['is_helpful'], tp)
            fAcc = pdAccuracy.loc[variant_name]
            
            print (f"Variant_name: {variant_name}, Accuracy: {fAcc:.4f}, ROAUC: {fAuc:.4f}, Precision: {fPrec:.4f}, Recall: {fRec:.4f}, FScore: {fScore:.4f}")
        
        # The metrics reported can change based on the model used, but it must be a specific name per 
        #(https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
        report_dict = {
            "binary_classification_metrics": {
                "accuracy": {"value": fAcc, "standard_deviation": "NaN",},
                "auc": {"value": fAuc, "standard_deviation": "NaN"},
                "prec": {"value": fPrec, "standard_deviation": "NaN"},
                "rec": {"value": fRec, "standard_deviation": "NaN"},
                "fscore": {"value": fScore, "standard_deviation": "NaN"},
            },
        }

        print("Classification report:\n{}".format(report_dict))

        evaluation_output_path = os.path.join(strOutputPath, "evaluation_" + self.args.variant_name.lower() + ".json")
        print("Saving classification report to {}".format(evaluation_output_path))

        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))

    def execution(self, ):
        
        if args.mode == 'docker':
            strModelPath = "s3://sagemaker-ap-northeast-2-419974056037/qrfa7iwq9a5f-HyperPa-g6aEWUAlNn-006-702449e1/output/model.tar.gz"
            strTestPath = "s3://sagemaker-ap-northeast-2-419974056037/preprocessing-2022-10-25-01-28-08-989/output/test_data/test.csv"
            strOutputPath = "/opt/ml/processing/evaluation"
            os.makedirs(strOutputPath)
        else:
            strModelPath = args.s3_model_path
            strTestPath = args.test_path
            strOutputPath = args.output_evaluation_dir
            print (f'isfile, Model: {os.path.isfile(strModelPath)}, Test: {os.path.isfile(strTestPath)}')
        print (f'strModelPath: {strModelPath}, \nstrTestPath: {strTestPath}, \nstrOutputPath: {strOutputPath}')
        
        strEndPointName = self._endpoint(strModelPath) ## Model artifact in s3 is only available (not in local(opy/ml/processing/model/) 
        #strEndPointName = "MAP-Inference-Endpoint-2022-10-26-01-50-03"
        listInferenceInput = self._preprocessing(strTestPath)
        pdPredResults = self._inference(listInferenceInput, strEndPointName, strTestPath)
        self._get_performance(pdPredResults, strOutputPath)
        self._release_resources(strEndPointName)

class eval_preprocess():
    
    def __init__(self, strDataPath):
        
        self.strDataPath = strDataPath
        self.index_to_label = {0: 'NotHelpful', 1: 'Helpful'} 
        
        nlp = English()
        self.tokenizer = nlp.tokenizer
        self.index_to_label = {0: 'NotHelpful', 1: 'Helpful'} 
    
    def _labelize_df(self, df):
        return '__label__' + df['is_helpful'].apply(lambda is_helpful: self.index_to_label[is_helpful])

    def _tokenize_sent(self, sent, max_length=1000):
        return ' '.join([token.text for token in self.tokenizer(sent)])[:max_length]

    def _tokenize_df(self, df):
        return (df['review_headline'].apply(self._tokenize_sent) + ' ' + 
                df['review_body'].apply(self._tokenize_sent))
    
    def execution(self, ):
        
        pdData = pd.read_csv(self.strDataPath)
        pdInput = self._tokenize_df(pdData).to_list()
        
        return pdInput
    
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

class clean_up():
    
    def __init__(self, ):    
        pass
    
    def delete_endpoint(self, client, endpoint_name ,is_del_model=True):
        
        response = client.describe_endpoint(EndpointName=endpoint_name)
        EndpointConfigName = response['EndpointConfigName']

        response = client.describe_endpoint_config(EndpointConfigName=EndpointConfigName)
        model_name = response['ProductionVariants'][0]['ModelName']    

        if is_del_model: # 모델도 삭제 여부 임.
            client.delete_model(ModelName=model_name)    

        client.delete_endpoint(EndpointName=endpoint_name)
        client.delete_endpoint_config(EndpointConfigName=EndpointConfigName)    

        print(f'--- Deleted model: {model_name}')
        print(f'--- Deleted endpoint: {endpoint_name}')
        print(f'--- Deleted endpoint_config: {EndpointConfigName}')  
        
        
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    parser.add_argument("--mode", type=str, default="docker")
    parser.add_argument('--model_path', type=str, default= "/opt/ml/processing/model/model.tar.gz")
    parser.add_argument('--s3_model_path', type=str, default= "s3://")
    parser.add_argument('--test_path', type=str, default= "/opt/ml/processing/test/test.csv")
    parser.add_argument('--output_evaluation_dir', type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument('--model_name', type=str, default= "helpfulness-detector-model")
    parser.add_argument('--variant_name', type=str, default= "a")
    parser.add_argument("--region", type=str, default="ap-northeast-2")
    parser.add_argument("--sagemaker_role", type=str, default="sagemaker_role")
    
    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    
    evaluation = evalauator(args)
    evaluation.execution() 