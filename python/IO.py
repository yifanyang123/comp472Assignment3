import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter

def read_training(filename):
    train=pd.read_csv('resource/'+filename+'.tsv',sep='\t')
    #train = pd.read_table('resource/'+filename,)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(train)
    dataText=train['text']
    dataLabel=train['q1_label']
    #result = [0 for x in range(dataLabel.size)]
    result=dict();
    for index,tweet in enumerate(dataText):
        tweet = re.sub('[^A-Za-z0-9-]', ' ',tweet).lower()   #only keep letter and space,and lower
        #nltk.download('punkt')   #if cannot tokenize, please use this!!!!!!!!!!!!!!!!!!!!!!!!
        words=nltk.word_tokenize(tweet)
        temp_dict=Counter(words)
        #print(temp_dict)
        for x in temp_dict:
            if (x in result):
                result[x][index]=temp_dict[x]
            else:
                result[x]=[0 for i in range(dataLabel.size)]
                result[x][index] = temp_dict[x]
    #nltk.download('stopwords')        #if cannot use stopwords, please use this!!!!!!!!!!!!!!!!!!!!!!!!
    for x in dict(result):
        if x in stopwords.words('english'):
            del result[x]
    originalResult=pd.DataFrame.from_dict(result)
    originalCSV=pd.concat([originalResult,dataLabel], axis=1)
    originalCSV.to_csv("output/"+filename+"_orginal.csv")

    for x in dict(result):
        if sum(result[x])<=1:
           del result[x]
    filterResult=pd.DataFrame.from_dict(result)
    filterCSV=pd.concat([filterResult,dataLabel], axis=1)
    filterCSV.to_csv("output/"+filename+"_filter.csv")

    return(originalResult,filterResult,dataLabel)

def read_test(filename):
    test=pd.read_csv('resource/'+filename+'.tsv',sep='\t',header=None)
    #train = pd.read_table('resource/'+filename,)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(train)
    dataID=test.iloc[:, 0]
    dataText=test.iloc[:,1]
    dataLabel=test.iloc[:,2]
    # #result = [0 for x in range(dataLabel.size)]
    result=dict();
    for index,tweet in enumerate(dataText):
        tweet = re.sub('[^A-Za-z0-9-]', ' ',tweet).lower()   #only keep letter and space,and lower
        #nltk.download('punkt')   #if cannot tokenize, please use this!!!!!!!!!!!!!!!!!!!!!!!!
        words=nltk.word_tokenize(tweet)
        temp_dict=Counter(words)
        #print(temp_dict)
        for x in temp_dict:
            if (x in result):
                result[x][index]=temp_dict[x]
            else:
                result[x]=[0 for i in range(dataLabel.size)]
                result[x][index] = temp_dict[x]
    #nltk.download('stopwords')        #if cannot use stopwords, please use this!!!!!!!!!!!!!!!!!!!!!!!!
    for x in dict(result):
        if x in stopwords.words('english'):
            del result[x]
    originalResult=pd.DataFrame.from_dict(result)
    #originalCSV=pd.concat([originalResult,dataLabel], axis=1)
    #originalCSV.to_csv("output/"+filename+"_orginal.csv")


    return(dataID,originalResult,dataLabel)


def generateOutput(fileName,tweetID,predictY,score,realY):
    trace=''
    total=0
    totalPredictYes=0
    totalRealYes=0
    totalPredictNo=0
    totalRealNo=0
    totalSuccess=0
    totalSuccessYes=0
    totalSuccessNo=0

    for i in range(0,len(tweetID)):
        trace+=f'{tweetID[i]}  {predictY[i]}  {score[i]}  {realY[i]}'
        if(predictY[i]==realY[i]):
            totalSuccess+=1
            trace += '  correct\n'
            if (predictY[i] == 'yes'):
                totalSuccessYes += 1
            else:
                totalSuccessNo += 1
        else:
            trace += '  wrong\n'
        if(predictY[i]=='yes'):
            totalPredictYes+=1
        else:
            totalPredictNo+=1
        if(realY[i]=='yes'):
            totalRealYes+=1
        else:
            totalRealNo+=1
        total+=1
    solutionoutput= open("output/trace_"+fileName+".txt", "w")
    solutionoutput.write(trace)

    accuracy=totalSuccess/total
    percisionY=totalSuccessYes/totalPredictYes
    percisionN=totalSuccessNo/totalPredictNo
    recallY=totalSuccessYes/totalRealYes
    recallN=totalSuccessNo/totalRealNo
    f1Y=2*percisionY*recallY/(percisionY+recallY)
    f1N=2*percisionN*recallN/(percisionN+recallN)
    evaluate=''
    evaluate+=f'{accuracy} \n'
    evaluate+=f'{percisionY}  {percisionN}\n'
    evaluate+=f'{recallY}  {recallN}\n'
    evaluate+=f'{f1Y}  {f1N}'
    solutionoutput= open("output/eval_"+fileName+".txt", "w")
    solutionoutput.write(evaluate)
