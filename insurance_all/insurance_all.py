class InsuranceAll(object):


    def feature_engineering(self, test):

        test['vehicle_age'] = test['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_year' if x== '1-2 Year' else 'under_1_year')

        test['vehicle_damage'] = test['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)

        return test
    

