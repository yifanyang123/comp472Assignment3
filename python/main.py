from IO import *
from Naive_Bayes import *




trainData=read_training('covid_training')
testData=read_test('covid_test_public')
trainOriginalX=trainData[0]
trainFilterX=trainData[1]
trainY=trainData[2]
testID=testData[0]
testX=testData[1]
testY=testData[2]


# nb=NaiveBayes()
# nb.fit(trainOriginalX,trainY)
# result=nb.predict(testX)
# predictY=result[0]
# scores=result[1]
# print(predictY)
# print(scores)
predictY=['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes']
scores=['-4.133540E+01', '-2.154584E+01', '-6.969858E+01', '-6.126145E+01', '-3.421577E+01', '-2.597609E+01', '-3.469686E+01', '-4.804065E+01', '-5.407544E+01', '-3.862444E+01', '-4.025032E+01', '-1.091451E+01', '-6.354770E+01', '-3.981385E+01', '-2.176693E+01', '-7.675448E+01', '-2.546992E+01', '-4.085089E+01', '-3.901140E+01', '-2.411697E+01', '-5.265140E+01', '-8.869302E+00', '-6.608819E+01', '-1.621008E+01', '-3.657813E+01', '-6.541997E+01', '-3.985539E+01', '-4.768427E+01', '-4.280882E+01', '-3.208301E+01', '-1.141522E+01', '-2.433509E+01', '-1.740596E+01', '-7.354978E+00', '-3.981385E+01', '-1.072609E+01', '-6.449968E+01', '-1.678857E+01', '-1.743898E+01', '-3.492143E+01', '-3.769159E+01', '-5.335761E+01', '-5.259433E+01', '-6.292950E+01', '-1.355269E+01', '-6.724628E+01', '-6.707015E+01', '-1.864766E+01', '-2.059548E+01', '-7.401687E+01', '-5.521968E+01', '-2.672940E+01', '-6.051036E+01', '-2.804398E+01', '-7.448769E+01']
generateOutput("NB-BOW-OV",testID,predictY,scores,testY)


# nb=NaiveBayes()
# nb.fit(trainFilterX,trainY)
# result=nb.predict(testX)
# predictY=result[0]
# scores=result[1]
# print(predictY)
# print(scores)
predictY=['yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes']
scores=['-3.574334E+01', '-2.041309E+01', '-5.146243E+01', '-4.399056E+01', '-2.515555E+01', '-1.356043E+01', '-3.299774E+01', '-4.230700E+01', '-4.109770E+01', '-2.755990E+01', '-2.557602E+01', '-1.016009E+01', '-5.307155E+01', '-3.047045E+01', '-1.565527E+01', '-5.863404E+01', '-2.405399E+01', '-3.179066E+01', '-2.980959E+01', '-1.450151E+01', '-3.219553E+01', '-8.416649E+00', '-5.786415E+01', '-1.515389E+01', '-2.404974E+01', '-5.508541E+01', '-2.518109E+01', '-3.097976E+01', '-2.438317E+01', '-1.769189E+01', '-1.066080E+01', '-1.977510E+01', '-1.285799E+01', '-6.902325E+00', '-3.047045E+01', '-6.549965E+00', '-5.402353E+01', '-1.070153E+01', '-1.254945E+01', '-2.301215E+01', '-2.315888E+01', '-3.997967E+01', '-3.188117E+01', '-4.912678E+01', '-1.264738E+01', '-5.490740E+01', '-4.075232E+01', '-1.404675E+01', '-1.946274E+01', '-6.339912E+01', '-4.891965E+01', '-1.623123E+01', '-4.498650E+01', '-2.318223E+01', '-6.372835E+01']
generateOutput("NB-BOW-FV",testID,predictY,scores,testY)