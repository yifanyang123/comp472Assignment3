import math

import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.prior=[0,0]
        self.y_conditional=dict()
        self.n_conditional=dict()

    def fit(self,x,y):
        y_counter=0
        n_counter=0
        for bool in y:
            if bool == 'yes':
                y_counter += 1
            elif bool == 'no':
                n_counter += 1
        self.prior[0]=y_counter/y.size
        self.prior[1]=n_counter/y.size
        totalwords=x.values.sum()
        print(totalwords)
        vocabulary=len(x.columns)
        trainTable = pd.concat([x, y], axis=1)
        totalYes=trainTable.query("q1_label=='yes'").sum(numeric_only=True).sum()
        totalNo=trainTable.query("q1_label=='no'").sum(numeric_only=True).sum()
        # print(totalYes)
        # print(totalNo)
        for column in x:
            # print (column)
            # print(x[column].sum())
            # print(trainTable.query("q1_label == 'yes'")[column].sum())
            # print(trainTable.query("q1_label == 'no'")[column].sum())
            # print(len(x.columns))
            self.y_conditional[column] = (trainTable.query("q1_label == 'yes'")[column].sum()+0.01)/(totalYes+vocabulary*0.01)
            self.n_conditional[column] = (trainTable.query("q1_label == 'no'")[column].sum()+0.01)/(totalNo+vocabulary*0.01)
        print("fit done")

    def predict(self,x):
        y_scores=[0 for i in range(len(x))]
        n_scores=[0 for i in range(len(x))]
        y=[]
        scores=[]
        for words in x:
            if(words in self.y_conditional):
                for index,wordsNum in enumerate(x[words]):
                     y_scores[index]+=wordsNum*math.log10(self.y_conditional[words])
                     n_scores[index]+=wordsNum*math.log10(self.n_conditional[words])
        print("predict 1 done")
        for index in range(0,len(y_scores)):
            y_scores[index] += math.log10(self.prior[0])
            n_scores[index] += math.log10(self.prior[1])
            if(y_scores[index]>=n_scores[index]):
                scores.append("{:E}".format(y_scores[index]))
                y.append("yes")
            else:
                scores.append("{:E}".format(n_scores[index]))
                y.append("no")
        return(y,scores)    
