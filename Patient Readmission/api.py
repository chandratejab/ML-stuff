###########################################################################################################
# How to use this API:
# -> Need to have cherrypy module
# -> Run this file. Engine serves on 127.0.0.1:8087
# -> Make POST request on 127.0.0.1:8087 with test json data in body. (Please find sample_input.json file) 
###########################################################################################################

import pandas as pd
from sklearn.externals import joblib
from datetime import datetime
import cherrypy

FILENAME = 'patient_readmission.pkl'

def process(input_json):
    unpick = joblib.load(FILENAME)
    model, labels_mapper, model_df_cols = unpick['model'], unpick['labels_mapper'], unpick['all_columns']

    # considered_cols = ['race', 'gender', 'age', 'admission_type_id', 'admission_source_id',
    #        'medical_specialty', 'num_procedures', 'num_medications',
    #        'num_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin', 'glipizide',
    #        'glyburide', 'insulin', 'change', 'diabetesMed', 'N_Days_Admitted',
    #        'admitted_in_season_1', 'admitted_in_season_2', 'admitted_in_season_3']

    dummy_cols = ['gender', 'change', 'diabetesMed', 'race', 'age', 'max_glu_serum', 
            'medical_specialty', 'A1Cresult', 'metformin', 'glipizide', 'glyburide', 'insulin', 
            'admission_type_id', 'admission_source_id']

    newob = {}

    newob['N_Days_Admitted'] = (datetime.strptime(input_json['Discharge_date'],'%Y-%m-%d')-datetime.strptime(input_json['Admission_date'],'%Y-%m-%d')).days

    admitted_month = datetime.strptime(input_json['Admission_date'],'%Y-%m-%d').month
    newob['admitted_in_season_1'] = 1 if admitted_month in [1,2,3,4] else 0
    newob['admitted_in_season_2'] = 1 if admitted_month in [5,6,7,8] else 0
    newob['admitted_in_season_3'] = 1 if admitted_month in [9,10,11,12] else 0

    for x in dummy_cols:
        if x == 'age':
            if input_json[x] in ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)']: input_json[x] = '[0-50)'
            elif input_json[x] in ['[80-90)', '[90-100)']: input_json[x] = '[80-100)'
                
        elif x == 'medical_specialty':
            if not input_json[x] in ['Cardiology', 'Emergency/Trauma', 'Family/GeneralPractice', 'InternalMedicine']: input_json[x] = 'other'
                
        newob[x] = labels_mapper[x][input_json[x]]

    for x in ['num_procedures', 'num_medications','num_diagnoses']:
        newob[x] = input_json[x]


    inp = pd.DataFrame([newob.values()],columns=newob.keys())
    inp = pd.get_dummies(inp,columns=dummy_cols)
    inp = inp.reindex(columns=model_df_cols,fill_value=0)

    print(inp.values)

    print('Actual value')
    output = 'Yes' if model.predict(inp) else 'No'
    print(output)
    print('Expected value')
    print(input_json['Target'])

    return output, input_json['Target']



class PRApi:
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def main(self):
        inp = cherrypy.request.json
        actual, expected = process(inp)

        print({'Actual Output': actual, 'Expected Output': expected})
        return {'Actual Output': actual, 'Expected Output': expected}
        
if __name__ == '__main__':
    try:
        cherrypy.server.socket_host = '127.0.0.1'
        cherrypy.server.socket_port = 8087
        cherrypy.response.timeout = 1500
        cherrypy.server.socket_timeout = 1500
        cherrypy.server.thread_pool = 20

        cherrypy.tree.mount(PRApi(), '/', config={})
        cherrypy.engine.start()
        cherrypy.engine.block()
    except Exception as e:
        logger.exception('Application failed to start - {}'.format(e))
        print("exception class")
