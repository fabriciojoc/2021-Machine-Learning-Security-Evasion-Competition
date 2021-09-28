from copy import deepcopy
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

# need for speed class
class NeedForSpeedModel():

    # numerical attributes
    NUMERICAL_ATTRIBUTES = [
        #'string_paths', 'string_urls', 'string_registry', 'string_MZ', 'size',
        'virtual_size', 'has_debug', 'imports', 'exports', 'has_relocations',
        'has_resources', 'has_signature', 'has_tls', 'symbols', 'timestamp', 
        'numberof_sections', 'major_image_version', 'minor_image_version', 
        'major_linker_version', 'minor_linker_version', 'major_operating_system_version',
        'minor_operating_system_version', 'major_subsystem_version', 
        'minor_subsystem_version', 'sizeof_code', 'sizeof_headers', 'sizeof_heap_commit'
    ]

    # categorical attributes
    CATEGORICAL_ATTRIBUTES = [
        'machine', 'magic'
    ]

    # textual attributes
    TEXTUAL_ATTRIBUTES = ['libraries', 'functions', 'exports_list',
                          'dll_characteristics_list', 'characteristics_list']

    # label
    LABEL = "label"

    # initialize NFS classifier
    def __init__(self, 
                categorical_extractor = OneHotEncoder(handle_unknown="ignore"), 
                # textual_extractor = TfidfVectorizer(max_features=500, token_pattern=r"(?<=\s)(.*?)(?=\s)"),
                textual_extractor = HashingVectorizer(n_features=50000, token_pattern=r"(?<=\s)(.*?)(?=\s)"),
                #feature_scaler = MinMaxScaler(),
                feature_scaler = MaxAbsScaler(),
                classifier = RandomForestClassifier()):
        self.base_categorical_extractor = categorical_extractor
        self.base_textual_extractor = textual_extractor
        self.base_feature_scaler = feature_scaler
        self.base_classifier = classifier

    # append features to original features list
    def _append_features(self, original_features, appended):
        if original_features:
            for l1, l2 in zip(original_features, appended):
                for i in l2:
                    l1.append(i)
            return(original_features)
        else:
            return appended.tolist()

    # train a categorical extractor
    def _train_categorical_extractor(self, categorical_attributes):
        # initialize categorical extractor
        self.categorical_extractor = deepcopy(self.base_categorical_extractor)
        # train categorical extractor
        self.categorical_extractor.fit(categorical_attributes.values)

    # transform categorical attributes into features
    def _transform_categorical_attributes(self, categorical_attributes):
        # transform categorical attributes using categorical extractor
        cat_features = self.categorical_extractor.transform(categorical_attributes.values)
        # return categorical features
        return cat_features

    # train a textual extractor
    def _train_textual_extractor(self, textual_attributes):
        # initialize textual extractors
        self.textual_extractors = {}
        # train feature extractor for each textual attribute
        for att in self.TEXTUAL_ATTRIBUTES:
            # initialize textual extractors
            self.textual_extractors[att] = deepcopy(self.base_textual_extractor)
            # train textual extractor
            self.textual_extractors[att].fit(textual_attributes[att].values)
    
    # transform textual extractor
    def _transform_textual_attributes(self, textual_attributes):
        # initialize features
        textual_features = None
        # extract features from each textual attribute
        for att in self.TEXTUAL_ATTRIBUTES:
            # train textual extractor
            att_features = self.textual_extractors[att].transform(textual_attributes[att].values)
            # transform into array (when it is an sparse matrix)
            # att_features = att_features.toarray()
            if textual_features == None:
                textual_features = att_features
            else:
                # append textual features
                textual_features = sparse.hstack((textual_features, att_features))
            # append textual features
            # textual_features = self._append_features(textual_features, att_features)
        return textual_features
        
    # train feature scaler
    def _train_feature_scaler(self, features):
        # initialize feature scaler
        self.feature_scaler = deepcopy(self.base_feature_scaler)
        # train feature scaler
        self.feature_scaler.fit(features)

    # transform features using feature scaler
    def _transform_feature_scaler(self, features):
        return self.feature_scaler.transform(features)

    # train classifier
    def _train_classifier(self,features,labels):
        # initialize classifier
        self.classifier = deepcopy(self.base_classifier)
        # train feature scaler
        self.classifier.fit(features, labels)

    # fit classifier using raw input
    def fit(self, train_data):
        # get labels
        train_labels = train_data[self.LABEL]
        # delete label column
        del train_data[self.LABEL]
        # initialize train_features with numerical ones
        train_features = sparse.csr_matrix(train_data[self.NUMERICAL_ATTRIBUTES].values)

        print("Training categorical features...", flush=True)
        # train categorical extractor
        self._train_categorical_extractor(train_data[self.CATEGORICAL_ATTRIBUTES])
        # transform categorical data
        cat_train_features = self._transform_categorical_attributes(train_data[self.CATEGORICAL_ATTRIBUTES])
        # append categorical_features to train_features
        # train_features = self._append_features(train_features, cat_train_features)
        train_features = sparse.hstack((train_features, cat_train_features))

        print("Training textual features...", flush=True)
        # train textual extractor (ALL DATA)
        self._train_textual_extractor(train_data[self.TEXTUAL_ATTRIBUTES])
        # train textual extractor (MALWARE ONLY)
        # self._train_textual_extractor(train_data[train_labels == 1][self.TEXTUAL_ATTRIBUTES])
        # transform textual data
        tex_train_features = self._transform_textual_attributes(train_data[self.TEXTUAL_ATTRIBUTES])
        # append textual_features to train_features
        # train_features = self._append_features(train_features, tex_train_features)
        train_features = sparse.hstack((train_features, tex_train_features))
        # transform in sparse matrix
        # train_features = csr_matrix(train_features)

        print("Normalizing features...", flush=True)
        # train feature normalizer
        self._train_feature_scaler(train_features)
        # transform features
        train_features = self._transform_feature_scaler(train_features)

        print("Training classifier...", flush=True)
        # train classifier
        return self._train_classifier(train_features, train_labels)


    def _extract_features(self,data):
        # initialize features with numerical ones
        # features = data[self.NUMERICAL_ATTRIBUTES].values.tolist()
        features = sparse.csr_matrix(data[self.NUMERICAL_ATTRIBUTES].values)

        print("Getting categorical features...", flush=True)
        # transform categorical data
        cat_features = self._transform_categorical_attributes(data[self.CATEGORICAL_ATTRIBUTES])
        # append categorical_features to features
        # features = self._append_features(features, cat_features)
        features = sparse.hstack((features, cat_features))

        print("Getting textual features...", flush=True)
        # transform textual data
        tex_features = self._transform_textual_attributes(data[self.TEXTUAL_ATTRIBUTES])
        # append textual_features to features
        # features = self._append_features(features, tex_features)
        features = sparse.hstack((features, tex_features))
        # transform in sparse matrix
        # features = csr_matrix(features)

        print("Normalizing features...", flush=True)
        # transform features
        features = self._transform_feature_scaler(features)

        # return features
        return(features)

    def predict(self,test_data):
        # extract features
        test_features = self._extract_features(test_data)        

        print("Predicting classes...", flush=True)
        # predict features
        return self.classifier.predict(test_features)

    def predict_proba(self,test_data):
        # extract features
        test_features = self._extract_features(test_data)        

        print("Predicting classes (proba)...", flush=True)
        # predict features
        return self.classifier.predict_proba(test_features)

    def predict_threshold(self,test_data, threshold=0.8):
        # extract features
        test_features = self._extract_features(test_data)        

        print("Predicting classes (threshold = {})...".format(threshold), flush=True)
        # predict features
        prob = self.classifier.predict_proba(test_features)
        # initialize pred
        pred = []
        # iterate over probabilities
        for p in prob:
            # add prediction
            pred.append(int(p[0] < threshold))
        # return prediction
        return pred