import gensim
import os
import json
from bs4 import BeautifulSoup
import numpy as np

class ModelEvaluator:
    def __init__(self, base_path="./new_filtering/lda_models"):
        self.base_path = base_path
        self.models = {}
        self.visualizations = {}
        
    def load_all_models(self):
        """Load all trained models from directories"""
        for model_dir in os.listdir(self.base_path):
            if model_dir.startswith('topics_'):
                model_path = os.path.join(self.base_path, model_dir)
                try:
                    # Load model components
                    model = gensim.models.LdaModel.load(
                        os.path.join(model_path, 'trained_model')
                    )
                    dictionary = gensim.corpora.Dictionary.load(
                        os.path.join(model_path, 'dictionary')
                    )
                    
                    vis_path = os.path.join(model_path, 'lda_visualization.html')
                    if os.path.exists(vis_path):
                        with open(vis_path, 'r') as f:
                            soup = BeautifulSoup(f, 'html.parser')
                            for script in soup.find_all('script'):
                                if 'var ldavis_el' in script.text:
                                    json_str = script.text.split('=')[1].strip().rstrip(';')
                                    vis_data = json.loads(json_str)
                                    self.visualizations[model_dir] = vis_data
                    
                    self.models[model_dir] = {
                        'model': model,
                        'dictionary': dictionary
                    }
                    print(f"Loaded model: {model_dir}")
                    
                except Exception as e:
                    print(f"Error loading {model_dir}: {str(e)}")
    
    def classify_text(self, text, preprocess_func):
        """
        Classify text using all loaded models
        Returns aggregated results with confidence scores
        """
        results = {}
        
        for model_name, model_data in self.models.items():
            model = model_data['model']
            dictionary = model_data['dictionary']
            
            processed_text = preprocess_func(text)
            bow = dictionary.doc2bow(processed_text)
            
            topic_dist = model.get_document_topics(bow)
            
            # Get top topic and its probability
            top_topic = max(topic_dist, key=lambda x: x[1])
            topic_id, prob = top_topic
            
            # Get topic terms
            topic_terms = model.show_topic(topic_id)
            
            results[model_name] = {
                'topic_id': topic_id,
                'probability': prob,
                'top_terms': [term for term, _ in topic_terms],
                'term_weights': [weight for _, weight in topic_terms]
            }
        
        return results
    
    def get_ensemble_prediction(self, text, preprocess_func, threshold=0.5):
        """
        Get ensemble prediction from all models
        Returns most confident prediction above threshold
        """
        classifications = self.classify_text(text, preprocess_func)
        
        # Filter by confidence threshold
        confident_predictions = {
            model: data for model, data in classifications.items()
            if data['probability'] > threshold
        }
        
        if not confident_predictions:
            return None
        
        # Return prediction with highest confidence
        best_model = max(
            confident_predictions.items(),
            key=lambda x: x[1]['probability']
        )
        
        return {
            'model': best_model[0],
            'prediction': best_model[1]
        }

# Usage example:
def preprocess_text(text):
    # Your preprocessing function here
    return text.lower().split()

# Initialize and load models
evaluator = ModelEvaluator("./new_filtering/lda_models")
evaluator.load_all_models()

# Classify new text
text = " 1.Article 2(6) of Directive 2011/83/EU of the European Parliament and of the Council of 25 October 2011 on consumer rights, amending Council Directive 93/13/EEC and Directive 1999/44/EC of the European Parliament and of the Council and repealing Council Directive 85/577/EEC and Directive 97/7/EC of the European Parliament and of the Council, read in conjunction with Article 3(1) of Directive 2011/83,must be interpreted as meaning that a leasing agreement relating to a motor vehicle, which is characterised by the fact that neither that agreement nor a separate agreement provides that the consumer is required to purchase the vehicle upon the expiry of the agreement, falls within the scope of Directive 2011/83, as a ‘service contract’ within the meaning of Article 2(6) thereof. By contrast, such an agreement does not fall within the scope of either Directive 2002/65/EC of the European Parliament and of the Council of 23 September 2002 concerning the distance marketing of consumer financial services and amending Council Directive 90/619/EEC and Directives 97/7/EC and 98/27/EC, or of Directive 2008/48/EC of the European Parliament and of the Council of 23 April 2008 on credit agreements for consumers and repealing Council Directive 87/102/EEC.2.Article 2(7) of Directive 2011/83must be interpreted as meaning that a service contract, within the meaning of Article 2(6) of that directive, concluded between a consumer and a trader by using a means of distance communication cannot be classified as a ‘distance contract’, within the meaning of the first of those provisions, where the conclusion of the contract was preceded by a negotiation stage which took place in the simultaneous physical presence of the consumer and an intermediary acting in the name or on behalf of the trader, and during which that consumer received from that intermediary, for the purposes of that negotiation, all the information referred to in Article 6 of that directive and was able to ask that intermediary questions about the proposed contract or offer in order to remove any uncertainty as to the scope of his or her possible contractual commitment with the trader.3.Article 2(8)(a) of Directive 2011/83must be interpreted as meaning that a service contract, within the meaning of Article 2(6) of that directive, concluded between a consumer and a trader, cannot be classified as an ‘off-premises contract’ within the meaning of the first of those provisions, where, during the stage preparing the ground for the conclusion of the contract through the use of a means of distance communication, the consumer visited the business premises of an intermediary acting in the name or on behalf of the trader for the purposes of the negotiation of that contract but operating in a field of activity other than that of the trader, provided that that consumer, as an average consumer who is reasonably well informed and reasonably observant and circumspect, could have expected, from visiting the business premises of the intermediary, to be solicited by that intermediary for the purposes of the negotiation and conclusion of a service contract with the trader and provided that that consumer could also easily have understood that that intermediary was acting in the name or on behalf of that trader.4.Article 16(l) of Directive 2011/83must be interpreted as meaning that a leasing agreement for a motor vehicle, concluded between a trader and a consumer and classified as a distance or off-premises service contract within the meaning of that directive, comes under the exception to the right of withdrawal laid down in that provision in respect of distance or off-premises contracts falling within the scope of that directive and concerning car rental services coupled with a specific date or period of performance, where the main purpose of that agreement is to allow the consumer to use a vehicle for the specific period of time stipulated in that agreement, in return for the regular payment of sums of money.5.Article 10(2)(p) of Directive 2008/48must be interpreted as precluding national legislation that establishes a statutory presumption that the trader is in compliance with its obligation to inform the consumer of his or her right of withdrawal where that trader refers, in a contract, to national provisions which themselves refer to a statutory information model in that regard, while using terms set out in that model which do not comply with the requirements of that provision of the directive. If it is not possible to interpret the national legislation at issue in a manner consistent with Directive 2008/48, a national court hearing a dispute exclusively between private individuals is not required, solely on the basis of EU law, to disapply such legislation, without prejudice to the possibility for that court to disapply it on the basis of its domestic law and, failing that, without prejudice to the right of the party harmed as a result of national law not being in conformity with EU law to claim compensation for the resulting loss which he or she has suffered.6.Article 10(2)(p) of Directive 2008/48, read in conjunction with Article 14(3)(b) of that directive,must be interpreted as meaning that the amount of daily interest that must be stated in a credit agreement pursuant to that provision, applicable where the consumer exercises the right of withdrawal, may not, under any circumstances, exceed the amount calculated from the contractual borrowing rate stipulated in that agreement. The information provided in the agreement concerning the amount of daily interest must be stated in a clear and concise manner so that, inter alia, read in conjunction with other information, it contains no contradiction objectively capable of misleading an average consumer who is reasonably well informed and reasonably observant and circumspect as to the amount of daily interest that he or she will ultimately have to pay. In the absence of such information, no amount of daily interest is payable.7.Article 10(2)(t) of Directive 2008/48must be interpreted as meaning that a credit agreement must specify the essential information concerning all out-of-court complaint or redress mechanisms available to the consumer and, where appropriate, the cost of using them, the fact that the complaint or application for redress must be submitted by post or electronically, the physical or electronic address to which that complaint or application for redress must be sent and the other formal conditions to which that complaint or application for redress is subject, since a mere reference, in the credit agreement, to rules of procedure available upon request or on the internet, or to another act or document concerning the detailed rules governing out-of-court complaint and redress mechanisms is insufficient.8.Article 10(2)(r) of Directive 2008/48must be interpreted as meaning that a credit agreement must, in principle, for the calculation of the compensation due in the event of early repayment of the loan, indicate the method of calculating that compensation in a manner that is specific and easily understandable for an average consumer who is reasonably well informed and reasonably observant and circumspect so that he or she can determine the amount of compensation due in the event of early repayment on the basis of the information provided in that agreement. That said, even in the absence of a specific and easily understandable indication of the method of calculation, such an agreement may satisfy the obligation set out in that provision provided that it contains other information enabling the consumer easily to determine the amount of the relevant compensation, in particular the maximum amount thereof, which he or she will have to pay in the event of early repayment of the loan.9.Point (b) of the second subparagraph of Article 14(1) of Directive 2008/48must be interpreted as meaning that, where information provided by the creditor to the consumer under Article 10(2) of that directive proves to be incomplete or incorrect, the withdrawal period starts to run only if the incomplete or incorrect nature of that information is not capable of affecting the consumer’s ability to assess the extent of his or her rights and obligations under that directive or his or her decision to conclude the contract and, where relevant, is not capable of depriving him or her of the possibility of exercising his or her rights, in essence, under the same conditions as would have prevailed if that information had been provided in a complete and correct manner.10.Article 10(2)(l) of Directive 2008/48must be interpreted as meaning that a credit agreement must state, as a specific percentage, the rate of late-payment interest that is applicable at the time the agreement is concluded and must describe in specific terms the mechanism for adjusting that rate. Where that rate is determined on the basis of a reference interest rate which varies over time, the credit agreement must state the reference interest rate that is applicable on the date the agreement is concluded, and the method for calculating the rate of late-payment interest on the basis of the reference interest rate must be set out in the agreement in a way which is readily understood by an average consumer who does not have specialist knowledge in the financial field, so that he or she can calculate the rate of late-payment interest based on the information provided in that agreement. Furthermore, the credit agreement must indicate the frequency with which that reference interest rate may be varied, even if that frequency is determined by national provisions.11.Article 14(1) of Directive 2008/48must be interpreted as meaning that the full performance of the credit agreement causes the right of withdrawal to be extinguished.Furthermore, the creditor cannot validly plead that, on account of the consumer’s conduct between the conclusion of the agreement and the exercise of the right of withdrawal, or even after exercising it, the consumer exercised that right abusively where, due to incomplete or incorrect information in the credit agreement, in breach of Article 10(2) of Directive 2008/48, the withdrawal period has not begun to run because it has been established that the incompleteness or incorrectness of that information affected the consumer’s ability to assess the extent of his or her rights and obligations under Directive 2008/48 and his or her decision to conclude the agreement.12.Directive 2008/48 must be interpreted as precluding a creditor from being able to plead, where the consumer exercises his or her right of withdrawal in accordance with Article 14(1) of that directive, that that right is time-barred under rules of national law, where at least one of the mandatory pieces of information referred to in Article 10(2) of that directive was not included in the credit agreement or was set out in it in an incomplete or incorrect manner without being duly communicated subsequently and where, on that ground, the withdrawal period provided for in Article 14(1) has not started to run.13.Article 14(1) of Directive 2008/48, read in conjunction with the principle of effectiveness,must be interpreted as precluding national legislation which provides that, where the consumer withdraws from a linked credit agreement, within the meaning of Article 3(n) of that directive, he or she must return to the creditor the goods financed by the credit or must have given the creditor formal notice to take back those goods without that creditor being required, at the same time, to repay the monthly instalments of the credit already paid by the consumer."
results = evaluator.get_ensemble_prediction(
    text,
    preprocess_text,
    threshold=0.3
)

if results:
    print(f"\nBest prediction from model: {results['model']}")
    print(f"Topic ID: {results['prediction']['topic_id']}")
    print(f"Confidence: {results['prediction']['probability']:.3f}")
    print("Top terms:", ', '.join(results['prediction']['top_terms'][:5]))
else:
    print("No confident predictions found")

# For Multiple Texts inputs , I want to ideally use a list , not fully sure

# Batch classification
# texts = [
#     "Courst Case1 judgment?",
#     "Case 2?",
#  
# ]

# Need to check whether as a list , checking each case separately works or as one text 

# for idx, text in enumerate(texts):
#     print(f"\nDocument {idx + 1}:")
#     result = evaluator.get_ensemble_prediction(text, preprocess_text)
#     if result:
#         print(f"Classified as Topic {result['prediction']['topic_id']}")
#         print(f"Confidence: {result['prediction']['probability']:.3f}")
#         print("Key terms:", ', '.join(result['prediction']['top_terms'][:3]))
