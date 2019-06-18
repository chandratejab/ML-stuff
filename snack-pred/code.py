print 'lol'
from sklearn import metrics
from sklearn import preprocessing
import cPickle
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import LinearSVC
from itertools import groupby
import os.path
import numpy as np
from optparse import OptionParser
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
import os
import warnings
warnings.filterwarnings("ignore")


def uncorr_df(x, corr_val):
    '''
    Obj: Drops features that are strongly correlated to other features.
          This lowers model complexity, and aids in generalizing the model.
    Inputs:
          df: features df (x)
          corr_val: Columns are dropped relative to the corr_val input (e.g. 0.8)
    Output: df that only includes uncorrelated features
    '''

    # Creates Correlation Matrix and Instantiates
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterates through Correlation Matrix Table to find correlated columns
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = item.values
            if val >= corr_val:
                # Prints the correlated feature set and the corr val
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(i)

    drops = sorted(set(drop_cols))[::-1]

    # Drops the correlated columns
    df=x.copy()
    for i in drops:
        col = x.iloc[:, (i+1):(i+2)].columns.values
        df = df.drop(col[0], axis=1)

    return df, drops, corr_matrix
def create_metrics_report(fname, testY, testY_pred, testY_prob):
#    f1=open("sklearn_results_test"+ str(_N_VARS) + "_" + str(_N_CLUSTERS) + ".txt" , 'w+')
    f1=open(fname , 'w+')

    print >>f1, metrics.classification_report(testY, testY_pred)
    # import pdb;pdb.set_trace()
    print >>f1,"auc=",  metrics.auc(testY, testY_pred, reorder=True)
    # print >>f1,"roc_auc", metrics.roc_auc_score(testY, testY_pred)    #*** ValueError: multiclass format is not supported
    print >>f1,"confusion_matrix", metrics.confusion_matrix(testY, testY_pred)    
    print >>f1,"accuracy_score",  metrics.accuracy_score(testY, testY_pred)    
    # print >>f1, "average precision - preds",metrics.average_precision_score(testY, testY_pred)    #ValueError: multiclass format is not supported
    # print >>f1, "average precision - prob-NoLease=0", metrics.average_precision_score(testY, testY_prob[:,0])   
    # print >>f1, "average precision - prob[-NoLease=1",metrics.average_precision_score(testY, testY_prob[:,1])   
    print >>f1, "f1_score", metrics.f1_score(testY, testY_pred)    
    # print >>f1, "logloss", metrics.log_loss(testY, testY_pred)    

    f1.close()
    return





# import pdb;pdb.set_trace()


def main(inpp, inp):

    clf_randomf = RandomForestClassifier(max_depth=10)
    clf_gradient = GradientBoostingClassifier()
    clf_linearsvc = LinearSVC()
    # import pdb;pdb.set_trace()
    data=pd.read_csv('snack.csv')
    data=data.drop(['working_sat'],axis=1)
    # train_df_all=data.drop(['working_sat','prev_snack_item','snack_item'],axis=1)

    # cat_dummies=pd.get_dummies(data[['working_sat','prev_snack_item']].copy() \
    # 						,['working_sat','prev_snack_item'])

    # import pdb;pdb.set_trace()


    Y=data['snack_item']

    le = preprocessing.LabelEncoder()

    types = ["aaku_pakodi", "appalu", "chips", "mixture", "murkulu","onion_pakodi", 
    "palli", "puffs", "punugulu", "samosa","soya","corn","corn_flakes","french_fries",
    "kachori","manchuria","chana","onion_bajji","wada","alu_bajji","corn_samosa","hotdog",
    "bajji","spring_roll"]

    le.fit(types)

    Y=le.transform(Y)
    data=data.drop(['snack_item'],axis=1)
    X=pd.get_dummies(data,columns=data.columns)

    # train_df_all=pd.concat([cat_dummies,train_df_all],axis=1)


    # uncorr,dropped_cols, all_corr_matrix=uncorr_df(train_df_all, 0.8)

    # X = uncorr.copy()


    trainX, testX, trainY, testY =  train_test_split(X, Y, test_size = .3, random_state = 166)

    # #X_F_std=X_F.loc[:,:].apply(lambda x: (x-x.mean())/x.std())
    # clf_randomf=linear_model.LogisticRegression(C=1.0,tol=.01, fit_intercept=True)
    # clf_randomf.fit(trainX, trainY)
    # print(clf_randomf.score(trainX,trainY))
    # print(trainY.mean())
    # print(clf_randomf.coef_)
    # print(clf_randomf.intercept_)

    # print(testY.mean())
    # X_F_pred=clf_randomf.predict(X_F)
    # X_F_pred_prob=clf_randomf.predict_proba(X_F)

    pd.set_option('display.max_columns', 50)

    clf_randomf.fit(X, Y) #Random forest

    clf_gradient.fit(X,Y)

    clf_linearsvc.fit(X,Y)
    # clf_randomf.fit(trainX, trainY) #Random forest

    # clf_gradient.fit(trainX ,trainY)

    outputs=[]
    #      [u'week', u'day', u'working_sat', u'prev_snack_item'])

    inp=pd.get_dummies(inp,columns=inp.columns)

    inp=inp.reindex(columns=X.columns,fill_value=0)

    print "Gradient Bossting"
    gb_pred=le.inverse_transform(clf_gradient.predict(inp))
    print(gb_pred)
    outputs.append(gb_pred)

    probs=pd.DataFrame(clf_gradient.predict_proba(inp))
    probs.columns=le.inverse_transform(clf_gradient.classes_)

    print probs


    # print(clf_gradient.score(testX,testY))

    print "Random Forest"
    rf_pred=le.inverse_transform(clf_randomf.predict(inp))
    print(rf_pred)
    outputs.append(rf_pred)

    probs=pd.DataFrame(clf_randomf.predict_proba(inp))
    probs.columns=le.inverse_transform(clf_randomf.classes_)
    print probs

    # print(clf_randomf.score(testX,testY))

    print "Linear SVC"
    linear_pred=le.inverse_transform(clf_linearsvc.predict(inp))
    print(linear_pred)
    outputs.append(linear_pred)

    # print(clf_gradient.score(testX,testY))

    print 'All Preds:'
    print outputs
    print '\n\nFinal prediction'

    def most_common(L):
      return max(groupby(sorted(L)), key=lambda(x, v):(len(list(v)),-L.index(x)))[0]
    print most_common([rf_pred,gb_pred,linear_pred])




    # X_F=X
    # X_F_pred=clf_randomf.predict(X_F)
    # X_F_pred_prob=clf_randomf.predict_proba(X_F)
    # fname="results.txt"


    # create_metrics_report(fname, Y, X_F_pred, X_F_pred_prob)





    snack_item_le = le
    data['prev_snack_item'] = snack_item_le.transform(data['prev_snack_item']) 

    real_le = preprocessing.LabelEncoder()
    real_le.fit(['yes','no'])
    data['real'] = real_le.transform(data['real'])

    day_name_le = preprocessing.LabelEncoder()
    day_name_le.fit(['monday','tuesday','wednesday','thursday','friday','saturday'])
    data['day_name'] = day_name_le.transform(data['day_name'])

    # Custom NN


    if os.path.isfile('nn.pkl'):
        with open('nn.pkl', 'rb') as model:
            model = cPickle.load(model)
            synapse_0 = model[0]
            synapse_1 = model[1]

    else: 
        # X = data.values
        ops=[]
        for i,x in enumerate(Y):
            # ops[i][x-1]=x
            op=[0]*len(types)
            op[x]=1
            ops.append(op)

        y = np.array(ops)

        H = 100 # number of hidden layer neurons

        alpha,hidden_dim = (0.02,H)#len(types))

        # Xavier initialization
        synapse_0 = np.random.random((39,hidden_dim)) / np.sqrt(39)
        synapse_1 = np.random.random((hidden_dim,len(types))) / np.sqrt(39)

        # Feature Scaling
        # from sklearn.preprocessing import StandardScaler
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X.values)
        # Random
        # synapse_0 = 2*np.random.random((39,hidden_dim)) - 1
        # synapse_1 = 2*np.random.random((hidden_dim,len(types))) - 1
        layer_0 = X.values
        for kk in range(60000):
            #Sigmoid
            # layer_1 = 1/(1+np.exp(-(np.dot(np.array([x]),synapse_0))))
            # layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
            
            #ReLU
            layer_1 = np.dot(layer_0,synapse_0)
            layer_1[layer_1<0] = 0
            
            layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))

            layer_2_error = layer_2 - y
            # lay2 = d-er * d-sgmL2
            layer_2_delta = 2*(layer_2_error)*(layer_2*(1-layer_2))
            # lay1 = lay2.syn1 * d-sgmL1
            layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
            synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
            synapse_0 -= (alpha * layer_0.T.dot(layer_1_delta))

            if (kk% 10000) == 0:
                # print 'example {}'.format(i)
                # error=0
                # for err in (layer_2-y[i])[0]:
                #     error += err**2
                # print 'error {}'.format(error)
                print "Error after "+str(kk)+" iterations:" + str(np.mean(np.abs(layer_2_error)))


        with open('nn.pkl', 'wb') as f:
            cPickle.dump([synapse_0, synapse_1], f, protocol=cPickle.HIGHEST_PROTOCOL)

    # (70x39).(39x100)=(70x100).(100x24)=(70x24)
    # (1x5).(5x23)=(1x23).(23x23)=(1x23)

    # 5   23   23
    
    layer_1 = 1/(1+np.exp(-(np.dot(inp,synapse_0))))
    ans = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))

    ans_index = ans[0].tolist().index(max(ans[0]))
    
    print 'NN pred'
    print le.inverse_transform(ans_index)
    print 'confidence {}'.format(max(ans[0]))
    print 'sum of conf {}'.format(sum(ans[0]))

    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 39))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(output_dim = len(types), init = 'uniform', activation = 'sigmoid'))

    # Compiling Neural Network
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # import pdb;pdb.set_trace()
    # Fitting our model 
    X=pd.get_dummies(data,columns=data.columns)
    Y = pd.get_dummies(Y)
    classifier.fit(X, Y, batch_size = 10, nb_epoch = 1000)
    ans_keras = classifier.predict(inp)

    ans_index_keras = ans_keras[0].tolist().index(max(ans_keras[0]))

    print 'Keras NN pred'
    print le.inverse_transform(ans_index_keras)
    print 'confidence {}'.format(max(ans_keras[0]))
    print 'sum of conf {}'.format(sum(ans_keras[0]))


    print 'All Preds:'
    print [rf_pred,gb_pred,linear_pred,le.inverse_transform(ans_index),le.inverse_transform(ans_index_keras)]
    print '\n\nFinal prediction'

    def most_common(L):
      return max(groupby(sorted(L)), key=lambda(x, v):(len(list(v)),-L.index(x)))[0]
    print most_common([rf_pred,gb_pred,linear_pred,le.inverse_transform(ans_index),le.inverse_transform(ans_index_keras)])

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--retrain", default=False)
    # parser.add_option("-q", "--quiet",
    #                   action="store_false", dest="verbose", default=True,
    #                   help="don't print status messages to stdout")
    (options, args) = parser.parse_args()

    if options.retrain and os.path.isfile('nn.pkl'): os.remove('nn.pkl')

    ####################################### Input for prediction ######################################################
    inpp=(3,3,"kachori","wednesday","yes")
    inp=pd.DataFrame([inpp],columns=[u'week', u'day', u'prev_snack_item',u'day_name','real'])
    ###################################################################################################################

    main(inpp, inp)
